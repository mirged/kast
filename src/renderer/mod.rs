//! The renderer: one frame of deferred PBR, start to finish.
//!
//! [`Renderer`] owns every pipeline, render target and bind group for the whole frame. The order
//! is fixed and each pass reads only targets the previous pass wrote:
//!
//! 1. **shadow** — render depth into the four cascade slices of one shadow map,
//! 2. **geometry** — write albedo, normal, emissive and reversed-Z depth (the G-buffer),
//! 3. **ssao** — half-resolution ambient occlusion from depth + normal, then a separable blur,
//! 4. **lighting** — full-screen deferred shade of the G-buffer: sky IBL, punctual lights,
//!    cascaded PCSS shadows, SSR, fog, and the environment as background,
//! 5. **dof** — optional, from the linearised depth,
//! 6. **transparent** — forward alpha-blended pass, back to front, into the lit HDR target,
//! 7. **bloom** — prefilter, downsample chain, upsample chain,
//! 8. **tonemap** — exposure, bloom mix, tonemap curve, vignette, gamma,
//! 9. **fxaa** — antialias into the present target (or the offscreen target when headless).
//!
//! Everything before the tonemap works in linear HDR; only the tonemap and FXAA passes know about
//! display encoding. Per-frame CPU work is limited to uniform uploads and a bounding-sphere cull,
//! so a static scene costs the same as a moving one.
//!
//! ### Typical use
//!
//! ```no_run
//! use kast::{camera::Camera, context::GpuContext, renderer::Renderer, scene};
//!
//! # fn main() -> anyhow::Result<()> {
//! let ctx = pollster::block_on(GpuContext::new_headless(1280, 720))?;
//! let mut scene = scene::default_scene(&ctx.device, &ctx.queue)?;
//! let mut renderer = Renderer::new(
//!     &ctx.device, &ctx.queue, 1280, 720,
//!     ctx.target_format, ctx.srgb_target, true, &mut scene,
//! )?;
//! let camera = Camera::default();
//! renderer.render(&ctx.device, &ctx.queue, &scene, &camera, None, 1.0 / 60.0);
//! # Ok(())
//! # }
//! ```

pub mod ibl;
pub mod shader;
pub mod shadow;
pub mod targets;

use anyhow::Result;
use glam::{Mat4, Vec3, Vec4};

use crate::camera::Camera;
use crate::gpu;
use crate::material::{self, create_material_bind_group, create_material_bind_group_layout};
use crate::scene::{LightKind, Scene};
use crate::types::*;

use ibl::Ibl;
use shadow::Shadow;
use targets::{BLOOM_LEVELS, Rt, Targets};

/// One entry in a draw list: what to draw, and where in the object buffer its data lives.
///
/// Draw lists are rebuilt from the scene whenever objects change (`update_objects`), then filtered
/// per frame by `cull`. `slot` is the index written into the per-instance vertex buffer, so the
/// same entry works for the geometry, shadow and transparent passes without re-sorting.
#[derive(Clone, Copy)]
struct Draw {
    /// Index into [`Scene::meshes`].
    mesh: usize,
    /// Index into [`Scene::materials`], used to bind the material's textures.
    material: usize,
    /// Index into the object storage buffer, passed to the shader as an instance attribute.
    slot: u32,
    /// World-space origin, used to sort the forward transparent pass back to front.
    origin: [f32; 3],
    /// World-space bounding sphere, for frustum and shadow-range culling.
    center: [f32; 3],
    radius: f32,
}

/// Every runtime-adjustable knob of the renderer.
///
/// Public and plain data: changing a field takes effect on the next `render` call with no pipeline
/// rebuild, no reallocation and no scene reload. Features are switched off by a `*_enabled` flag
/// (the pass is skipped entirely, so a disabled effect costs nothing but the target that was
/// already allocated).
pub struct Settings {
    /// Exposure offset in stops, applied in the tonemap pass. `Camera::exposure_ev` is the usual
    /// source; positive brightens.
    pub exposure_ev: f32,
    /// How much of the bloom chain is mixed back in. 0 disables the effect visually without
    /// skipping the passes.
    pub bloom_intensity: f32,
    /// HDR luminance above which a pixel starts to bloom.
    pub bloom_threshold: f32,
    /// Width of the soft threshold ramp, so bloom fades in instead of popping at the threshold.
    pub bloom_knee: f32,
    /// Whether screen-space ambient occlusion is sampled by the lighting pass.
    pub ssao_enabled: bool,
    /// Sampling radius in view-space metres — roughly the size of the creases it darkens.
    pub ssao_radius: f32,
    /// Multiplier on the occlusion term; 0 is no darkening, values above 1 exaggerate it.
    pub ssao_intensity: f32,
    /// Depth offset that prevents self-occlusion acne on flat surfaces.
    pub ssao_bias: f32,
    /// Whether the sun casts shadows at all.
    pub shadows_enabled: bool,
    /// How dark a fully shadowed fragment is: 1 is black shadow, 0 leaves only the sky's IBL.
    pub shadow_strength: f32,
    /// Penumbra growth of sun shadows: metres of blur per metre of gap between blocker and
    /// receiver, i.e. how large the sun is treated as. 0 leaves a fixed one-texel filter.
    pub shadow_softness: f32,
    /// Whether the prefiltered sky cubemaps light the scene.
    pub ibl_enabled: bool,
    /// Multiplier on ambient and reflection energy from the IBL; useful when a sky is too dim or
    /// too bright but the direct sun should stay as it is.
    pub ibl_intensity: f32,
    /// Whether screen-space reflections are traced at all.
    pub ssr_enabled: bool,
    /// How strongly reflections are mixed over the shaded colour.
    pub ssr_intensity: f32,
    /// Maximum ray-march steps per reflection ray; more steps means longer range and more cost.
    pub ssr_steps: f32,
    /// Whether the depth-of-field pass runs.
    pub dof_enabled: bool,
    /// Focus distance in metres from the eye.
    pub dof_focus: f32,
    /// Half-width of the in-focus band around `dof_focus`, in metres; outside it blur ramps up.
    pub dof_range: f32,
    /// Largest blur radius in pixels, which caps the cost of the gather.
    pub dof_max_radius: f32,
    /// Whether FXAA runs as the last pass. Turning it off is useful when inspecting aliasing.
    pub fxaa_enabled: bool,
    /// Tonemap curve: 0 = ACES (Narkowicz), 1 = ACES (fitted), 2 = Reinhard, 3 = clamped linear.
    pub tonemap_mode: u32,
    /// Vignette strength; 0 leaves corners untouched, 1 darkens them fully.
    pub vignette: f32,
    /// Multiplier on the ambient term added on top of the IBL, for scenes with no sky.
    pub ambient: f32,
    /// Show the raw SSAO buffer instead of shading — the fastest way to debug radius and bias.
    pub debug_ssao: bool,
    /// Far edge of the cascaded shadow range (clamped to camera far plane).
    pub shadow_distance: f32,
    /// Whether height fog / aerial perspective is applied in the lighting pass.
    pub fog_enabled: bool,
    /// Extinction coefficient at sea level, per metre.
    pub fog_density: f32,
    /// Exponential height falloff of the fog, per metre.
    pub fog_height_falloff: f32,
    /// Hard ceiling on how much of the view the fog can hide, so distant geometry never vanishes
    /// completely.
    pub fog_max_opacity: f32,
    /// Extra in-scattered sun glow added to the fog colour toward the sun.
    pub fog_sun_glow: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            exposure_ev: 0.0,
            bloom_intensity: 0.12,
            bloom_threshold: 1.0,
            bloom_knee: 0.6,
            ssao_enabled: true,
            ssao_radius: 0.6,
            ssao_intensity: 1.1,
            ssao_bias: 0.025,
            shadows_enabled: true,
            shadow_strength: 1.0,
            shadow_softness: 0.02,
            ibl_enabled: true,
            ibl_intensity: 1.0,
            ssr_enabled: true,
            ssr_intensity: 0.55,
            ssr_steps: 48.0,
            dof_enabled: false,
            dof_focus: 8.0,
            dof_range: 12.0,
            dof_max_radius: 12.0,
            fxaa_enabled: true,
            tonemap_mode: 0,
            vignette: 0.35,
            ambient: 0.25,
            debug_ssao: false,
            shadow_distance: f32::INFINITY,
            fog_enabled: false,
            fog_density: 0.00004,
            fog_height_falloff: 0.00012,
            fog_max_opacity: 0.97,
            fog_sun_glow: 0.0,
        }
    }
}

struct TargetBindGroups {
    ssao: wgpu::BindGroup,
    ssao_blur: wgpu::BindGroup,
    lighting_g1: wgpu::BindGroup,
    ssr: wgpu::BindGroup,
    dof: wgpu::BindGroup,
    bloom_down: Vec<wgpu::BindGroup>,
    bloom_up: Vec<wgpu::BindGroup>,
    tonemap: wgpu::BindGroup,
    fxaa: wgpu::BindGroup,
}

/// The renderer: all pipelines, targets and per-frame state for one view.
///
/// Create it once with [`Renderer::new`] (which also uploads the scene), then call
/// [`Renderer::render`] per frame and [`Renderer::resize`] when the drawable changes size. The
/// pipeline is built for the target format it is created with, so a renderer cannot be moved
/// between an sRGB and a non-sRGB surface without rebuilding it.
pub struct Renderer {
    /// The G-buffer, HDR ping-pong targets, LDR target, SSAO buffers and the bloom chain.
    pub targets: Targets,
    /// The cascade shadow map and its per-frame uniform buffer.
    pub shadow: Shadow,
    /// The baked environment: irradiance cube plus prefiltered radiance chain. Rebuilt by
    /// [`Renderer::refresh_environment`].
    #[allow(dead_code)]
    pub ibl: Ibl,
    /// Every runtime-adjustable setting; see [`Settings`].
    pub settings: Settings,
    /// Format of the final target: the surface format when windowed, `Rgba8Unorm` when headless.
    pub present_format: wgpu::TextureFormat,
    /// Whether `present_format` already applies gamma, i.e. whether the shaders must skip it.
    pub srgb_target: bool,
    /// The offscreen present target, allocated only in headless mode. Read it back after
    /// `render` to save a screenshot.
    pub present: Option<Rt>,

    globals_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    lighting_uniform_buffer: wgpu::Buffer,
    sky_uniform_buffer: wgpu::Buffer,
    ssao_uniform_buffer: wgpu::Buffer,
    ssr_uniform_buffer: wgpu::Buffer,
    post_uniform_buffer: wgpu::Buffer,
    dof_uniform_buffer: wgpu::Buffer,
    fxaa_uniform_buffer: wgpu::Buffer,
    bloom_down_uniforms: Vec<wgpu::Buffer>,
    bloom_up_uniforms: Vec<wgpu::Buffer>,

    objects_buffer: wgpu::Buffer,
    object_index_buffer: wgpu::Buffer,
    objects_bind_group: wgpu::BindGroup,
    objects_bgl: wgpu::BindGroupLayout,
    draws: Vec<Draw>,
    shadow_draws: Vec<Draw>,
    transparent_draws: Vec<Draw>,
    /// `draws` / `shadow_draws` after culling against the current camera.
    visible_draws: Vec<Draw>,
    visible_shadow_draws: Vec<Draw>,

    #[allow(dead_code)]
    geometry_globals_bgl: wgpu::BindGroupLayout,
    geometry_globals_bg: wgpu::BindGroup,
    material_bgl: wgpu::BindGroupLayout,
    lighting_g0: wgpu::BindGroup,
    lighting_g0_layout: wgpu::BindGroupLayout,

    color_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,

    geometry_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    ssao_blur_pipeline: wgpu::RenderPipeline,
    lighting_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    ssr_pipeline: wgpu::RenderPipeline,
    bloom_prefilter: wgpu::RenderPipeline,
    bloom_downsample: wgpu::RenderPipeline,
    bloom_upsample: wgpu::RenderPipeline,
    dof_pipeline: wgpu::RenderPipeline,
    tonemap_pipeline: wgpu::RenderPipeline,
    fxaa_pipeline: wgpu::RenderPipeline,

    bgs: TargetBindGroups,
    time: f32,
}

impl Renderer {
    /// Build every pipeline and render target, and upload `scene`.
    ///
    /// `present_format`/`srgb_target` come from [`crate::context::GpuContext`] and must match the
    /// surface the frame will be presented to. `headless` allocates an internal `present` target
    /// (readable with `COPY_SRC`) instead of expecting a swap-chain view, and `scene` is taken by
    /// mutable reference because material bind groups are written back into it.
    ///
    /// This is the expensive call in the crate: it compiles every WGSL pipeline and generates the
    /// procedural IBL. Do it once per renderer, not per frame.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        present_format: wgpu::TextureFormat,
        srgb_target: bool,
        headless: bool,
        scene: &mut Scene,
    ) -> Result<Self> {
        let settings = Settings::default();
        let targets = Targets::new(device, width, height);
        let shadow = Shadow::new(device, shadow::SHADOW_SIZE);

        let sky_uniform = SkyUniform {
            sun_direction: scene.sky.sun_direction.extend(1.0).into(),
            sun_color: (scene.sky.sun_color * scene.sky.sun_intensity)
                .extend(1.0)
                .into(),
            params: [scene.sky.turbidity, scene.sky.exposure, 0.0, 0.0],
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
            zenith: scene
                .sky
                .zenith_color
                .extend(scene.sky.horizon_falloff)
                .into(),
            horizon: scene.sky.horizon_color.extend(1.0).into(),
        };
        let ibl = Ibl::new(device, queue, &sky_uniform);

        let color_sampler = material::create_sampler(device, "color-sampler", 16);
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("point-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let globals_buffer = gpu::uniform_buffer(device, "globals", &Globals::default());
        let lights_buffer = gpu::buffer_empty(
            device,
            "lights",
            (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let lighting_uniform_buffer = gpu::uniform_buffer(
            device,
            "lighting-u",
            &LightingUniform {
                params: [1.0, 1.0, 1.0, 1.0],
                params2: [1.0, (ibl::PREFILTER_MIPS - 1) as f32, 0.0, 1.0],
                fog: [0.0; 4],
                fog_sun: [0.0; 4],
            },
        );
        let sky_uniform_buffer = gpu::uniform_buffer(device, "sky-u", &sky_uniform);
        let ssao_uniform_buffer = gpu::uniform_buffer(
            device,
            "ssao-u",
            &SsaoUniform {
                proj: Mat4::IDENTITY.to_cols_array_2d(),
                inv_proj: Mat4::IDENTITY.to_cols_array_2d(),
                view: Mat4::IDENTITY.to_cols_array_2d(),
                inv_view: Mat4::IDENTITY.to_cols_array_2d(),
                params: [1.0; 4],
                screen: [16.0, width as f32, height as f32, 1.0],
            },
        );
        let ssr_uniform_buffer = gpu::uniform_buffer(
            device,
            "ssr-u",
            &SsrUniform {
                proj: Mat4::IDENTITY.to_cols_array_2d(),
                inv_proj: Mat4::IDENTITY.to_cols_array_2d(),
                view: Mat4::IDENTITY.to_cols_array_2d(),
                params: [64.0, 0.2, 0.08, 1.0],
                screen: [width as f32, height as f32, 0.0, 40.0],
            },
        );
        let post_uniform_buffer = gpu::uniform_buffer(
            device,
            "post-u",
            &PostUniform {
                params: [1.0, 2.2, 0.1, 0.3],
                modes: [0.0, 1.0, 1.0, if srgb_target { 0.0 } else { 1.0 }],
                screen: [
                    width as f32,
                    height as f32,
                    1.0 / width as f32,
                    1.0 / height as f32,
                ],
            },
        );
        let dof_uniform_buffer = gpu::uniform_buffer(
            device,
            "dof-u",
            &DofUniform {
                inv_proj: Mat4::IDENTITY.to_cols_array_2d(),
                params: [8.0, 12.0, 10.0, 1.0],
                depth: [0.05, 500.0, width as f32, height as f32],
            },
        );
        let fxaa_uniform_buffer = gpu::uniform_buffer(
            device,
            "fxaa-u",
            &FxaaUniform {
                params: [1.0 / width as f32, 1.0 / height as f32, 1.0, 0.0],
            },
        );

        let bloom_down_uniforms: Vec<wgpu::Buffer> = (0..BLOOM_LEVELS)
            .map(|i| {
                gpu::uniform_buffer(
                    device,
                    &format!("bloom-down-{i}"),
                    &BloomUniform {
                        params: [0.0; 4],
                        screen: [0.0; 4],
                    },
                )
            })
            .collect();
        let bloom_up_uniforms: Vec<wgpu::Buffer> = (0..BLOOM_LEVELS)
            .map(|i| {
                gpu::uniform_buffer(
                    device,
                    &format!("bloom-up-{i}"),
                    &BloomUniform {
                        params: [0.0; 4],
                        screen: [0.0; 4],
                    },
                )
            })
            .collect();

        let material_bgl = create_material_bind_group_layout(device);
        let geometry_globals_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("geometry-globals-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Globals>() as u64
                        ),
                    },
                    count: None,
                }],
            });
        let geometry_globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("geometry-globals-bg"),
            layout: &geometry_globals_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let objects_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("objects-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ObjectData>() as u64
                    ),
                },
                count: None,
            }],
        });

        let objects_buffer = gpu::buffer_empty(
            device,
            "objects",
            (std::mem::size_of::<ObjectData>().max(16)) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let object_index_buffer = gpu::buffer_empty(
            device,
            "object-indices",
            16,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        );
        let objects_bind_group = create_objects_bind_group(device, &objects_bgl, &objects_buffer);

        let geometry_pipeline =
            create_geometry_pipeline(device, &geometry_globals_bgl, &objects_bgl, &material_bgl);
        let shadow_pipeline = create_shadow_pipeline(device, &shadow.layout, &objects_bgl);

        let ssao_pipeline = create_screen_pipeline(
            device,
            &create_ssao_layout(device),
            include_str!("../../shaders/ssao.wgsl"),
            "fs_ssao",
            wgpu::TextureFormat::R8Unorm,
            wgpu::BlendState::REPLACE,
        );
        let ssao_blur_layout = create_ssao_layout(device);
        let ssao_blur_pipeline = create_screen_pipeline(
            device,
            &ssao_blur_layout,
            include_str!("../../shaders/ssao.wgsl"),
            "fs_blur",
            wgpu::TextureFormat::R8Unorm,
            wgpu::BlendState::REPLACE,
        );

        let lighting_g0_layout = create_lighting_layout_g0(device);
        let lighting_g1_layout = create_lighting_layout_g1(device);
        let lighting_pipeline = create_screen_pipeline_multi(
            device,
            &[&lighting_g0_layout, &lighting_g1_layout],
            include_str!("../../shaders/lighting.wgsl"),
            "fs_main",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::BlendState::REPLACE,
        );

        let transparent_pipeline =
            create_transparent_pipeline(device, &lighting_g0_layout, &objects_bgl, &material_bgl);

        let ssr_pipeline = create_screen_pipeline(
            device,
            &create_ssr_layout(device),
            include_str!("../../shaders/ssr.wgsl"),
            "fs_main",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::BlendState::REPLACE,
        );

        let bloom_downsample = create_screen_pipeline(
            device,
            &create_bloom_layout(device),
            include_str!("../../shaders/bloom.wgsl"),
            "fs_downsample",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::BlendState::REPLACE,
        );
        let bloom_prefilter = create_screen_pipeline(
            device,
            &create_bloom_layout(device),
            include_str!("../../shaders/bloom.wgsl"),
            "fs_prefilter",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::BlendState::REPLACE,
        );
        let bloom_upsample = create_screen_pipeline(
            device,
            &create_bloom_layout(device),
            include_str!("../../shaders/bloom.wgsl"),
            "fs_upsample",
            wgpu::TextureFormat::Rgba16Float,
            additive_blend(),
        );

        let dof_pipeline = create_screen_pipeline(
            device,
            &create_dof_layout(device),
            include_str!("../../shaders/dof.wgsl"),
            "fs_main",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::BlendState::REPLACE,
        );

        let tonemap_pipeline = create_screen_pipeline(
            device,
            &create_tonemap_layout(device),
            include_str!("../../shaders/tonemap.wgsl"),
            "fs_main",
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::BlendState::REPLACE,
        );

        let fxaa_pipeline = create_screen_pipeline(
            device,
            &create_fxaa_layout(device),
            include_str!("../../shaders/fxaa.wgsl"),
            "fs_main",
            present_format,
            wgpu::BlendState::REPLACE,
        );

        let present = if headless {
            Some(Rt::new(
                device,
                "present",
                width,
                height,
                present_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                1,
            ))
        } else {
            None
        };

        let lighting_g0 = build_lighting_g0(
            device,
            &lighting_g0_layout,
            &globals_buffer,
            &lights_buffer,
            &shadow,
            &ibl,
            &lighting_uniform_buffer,
        );

        let bgs = build_target_bindings(
            device,
            &targets,
            &ssao_uniform_buffer,
            &ssr_uniform_buffer,
            &dof_uniform_buffer,
            &post_uniform_buffer,
            &fxaa_uniform_buffer,
            &bloom_down_uniforms,
            &bloom_up_uniforms,
            &color_sampler,
            &point_sampler,
        );

        let mut renderer = Self {
            targets,
            shadow,
            ibl,
            settings,
            present_format,
            srgb_target,
            present,
            globals_buffer,
            lights_buffer,
            lighting_uniform_buffer,
            sky_uniform_buffer,
            ssao_uniform_buffer,
            ssr_uniform_buffer,
            post_uniform_buffer,
            dof_uniform_buffer,
            fxaa_uniform_buffer,
            bloom_down_uniforms,
            bloom_up_uniforms,
            objects_buffer,
            object_index_buffer,
            objects_bind_group,
            objects_bgl,
            draws: Vec::new(),
            shadow_draws: Vec::new(),
            transparent_draws: Vec::new(),
            visible_draws: Vec::new(),
            visible_shadow_draws: Vec::new(),
            geometry_globals_bgl,
            geometry_globals_bg,
            material_bgl,
            lighting_g0,
            lighting_g0_layout,
            color_sampler,
            point_sampler,
            geometry_pipeline,
            shadow_pipeline,
            ssao_pipeline,
            ssao_blur_pipeline,
            lighting_pipeline,
            transparent_pipeline,
            ssr_pipeline,
            bloom_prefilter,
            bloom_downsample,
            bloom_upsample,
            dof_pipeline,
            tonemap_pipeline,
            fxaa_pipeline,
            bgs,
            time: 0.0,
        };

        renderer.upload_scene(device, queue, scene);
        renderer.update_bloom_uniforms(queue);

        Ok(renderer)
    }

    /// Re-upload everything a scene owns and rebuild its material bind groups.
    ///
    /// Use this after adding meshes or materials (which allocate GPU resources); for transforms,
    /// visibility and parameter tweaks, [`Renderer::update_objects`] is enough and much cheaper.
    pub fn upload_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &mut Scene) {
        self.update_objects(device, queue, scene);
        log::info!(
            "uploaded scene: {} meshes, {} materials, {} objects, {} draws, object_buffer={} bytes",
            scene.meshes.len(),
            scene.materials.len(),
            scene.objects.len(),
            self.draws.len(),
            self.objects_buffer.size()
        );
        for mat in &mut scene.materials {
            mat.bind_group = Some(create_material_bind_group(
                device,
                &self.material_bgl,
                mat,
                &self.color_sampler,
            ));
        }
    }

    /// Re-bake the image-based lighting from the scene's current sky. Call this
    /// when the time of day or weather changes the sky, so a night city is lit
    /// by a night sky instead of the one the renderer started with.
    pub fn refresh_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
    ) {
        let sky_uniform = SkyUniform {
            sun_direction: scene.sky.sun_direction.extend(1.0).into(),
            sun_color: (scene.sky.sun_color * scene.sky.sun_intensity)
                .extend(1.0)
                .into(),
            params: [scene.sky.turbidity, scene.sky.exposure, 0.0, 0.0],
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
            zenith: scene
                .sky
                .zenith_color
                .extend(scene.sky.horizon_falloff)
                .into(),
            horizon: scene.sky.horizon_color.extend(1.0).into(),
        };
        self.ibl = Ibl::new(device, queue, &sky_uniform);
        queue.write_buffer(
            &self.sky_uniform_buffer,
            0,
            bytemuck::bytes_of(&sky_uniform),
        );
        self.lighting_g0 = build_lighting_g0(
            device,
            &self.lighting_g0_layout,
            &self.globals_buffer,
            &self.lights_buffer,
            &self.shadow,
            &self.ibl,
            &self.lighting_uniform_buffer,
        );
    }

    /// Re-uploads per-object transforms, material parameters and visibility.
    ///
    /// This is the per-frame path for animated scenes: it rewrites the object storage buffer, the
    /// per-instance slot indices and the three draw lists, but never touches pipelines or textures.
    /// Materials whose bind groups have not been built yet are left unbound and simply skipped by
    /// the draw loops.
    /// Cheap enough to call every frame for animated scenes; does not rebuild
    /// material bind groups (call `upload_scene` after adding materials).
    pub fn update_objects(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene) {
        let mut object_data = Vec::with_capacity(scene.objects.len());
        let mut draws = Vec::with_capacity(scene.objects.len());
        let mut shadow_draws = Vec::with_capacity(scene.objects.len());
        let mut transparent_draws = Vec::new();

        for obj in scene.objects.iter() {
            if !obj.visible {
                continue;
            }
            let mat = &scene.materials[obj.material];
            let normal_matrix = obj.transform.inverse().transpose();
            let d = &mat.desc;

            let dither = d.dithered || obj.alpha < 0.999;
            object_data.push(ObjectData {
                model: obj.transform.to_cols_array_2d(),
                normal_matrix: normal_matrix.to_cols_array_2d(),
                base_color: [
                    d.base_color[0],
                    d.base_color[1],
                    d.base_color[2],
                    d.base_color[3] * obj.alpha,
                ],
                emissive: [
                    d.emissive[0],
                    d.emissive[1],
                    d.emissive[2],
                    d.emissive_strength,
                ],
                material: [d.metallic, d.roughness, d.ao, d.alpha_cutoff],
                uv_transform: [d.uv_scale[0], d.uv_scale[1], d.uv_offset[0], d.uv_offset[1]],
                flags0: [
                    mat.has_albedo as u32,
                    mat.has_normal as u32,
                    mat.has_mr as u32,
                    mat.has_emissive as u32,
                ],
                flags1: [
                    mat.has_occlusion as u32,
                    d.unlit as u32,
                    d.double_sided as u32,
                    dither as u32,
                ],
                detail: [d.detail_scale, d.detail_strength, d.detail_anim, 0.0],
            });
            let origin = [
                obj.transform.w_axis.x,
                obj.transform.w_axis.y,
                obj.transform.w_axis.z,
            ];
            let (center, radius) = {
                let m = &scene.meshes[obj.mesh];
                let c = obj.transform.transform_point3((m.min + m.max) * 0.5);
                let scale = obj
                    .transform
                    .x_axis
                    .truncate()
                    .length()
                    .max(obj.transform.y_axis.truncate().length())
                    .max(obj.transform.z_axis.truncate().length());
                ([c.x, c.y, c.z], (m.max - m.min).length() * 0.5 * scale)
            };
            // Really transparent materials go to the forward pass; the deferred path can only
            // dither them.
            let draw = Draw {
                mesh: obj.mesh,
                material: obj.material,
                // Slot indexes the packed (visible-only) object buffer.
                slot: object_data.len() as u32 - 1,
                origin,
                center,
                radius,
            };
            if d.transparent || obj.alpha < 0.999 {
                transparent_draws.push(draw);
                continue;
            }
            if obj.cast_shadows {
                shadow_draws.push(draw);
            }
            draws.push(draw);
        }

        let object_bytes = bytemuck::cast_slice(&object_data);
        let needed = (object_bytes.len() as u64).max(16);
        if needed > self.objects_buffer.size() {
            self.objects_buffer = gpu::buffer_empty(
                device,
                "objects",
                needed,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            self.objects_bind_group =
                create_objects_bind_group(device, &self.objects_bgl, &self.objects_buffer);
        }
        if !object_data.is_empty() {
            queue.write_buffer(&self.objects_buffer, 0, object_bytes);
        }

        let indices: Vec<u32> = (0..object_data.len() as u32).collect();
        let index_bytes: &[u8] = bytemuck::cast_slice(&indices);
        let index_needed = (index_bytes.len() as u64).max(16);
        if index_needed > self.object_index_buffer.size() {
            self.object_index_buffer = gpu::buffer_empty(
                device,
                "object-indices",
                index_needed,
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            );
        }
        if !indices.is_empty() {
            queue.write_buffer(&self.object_index_buffer, 0, bytemuck::cast_slice(&indices));
        }

        self.draws = draws;
        self.shadow_draws = shadow_draws;
        self.transparent_draws = transparent_draws;
    }

    /// Reallocate every size-dependent target after the drawable changed size.
    ///
    /// No-op when the size is unchanged, so it is safe to call every frame. The G-buffer, HDR and
    /// bloom targets, the SSAO buffers and the offscreen present target are recreated, and the
    /// bloom and post uniforms are rewritten for the new extents; pipelines are untouched.
    /// Pair it with [`crate::context::GpuContext::resize`], which handles the surface itself.
    pub fn resize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.targets.size == (width, height) {
            return;
        }
        self.targets = Targets::new(device, width, height);
        if self.present.is_some() {
            self.present = Some(Rt::new(
                device,
                "present",
                width,
                height,
                self.present_format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                1,
            ));
        }
        self.bgs = build_target_bindings(
            device,
            &self.targets,
            &self.ssao_uniform_buffer,
            &self.ssr_uniform_buffer,
            &self.dof_uniform_buffer,
            &self.post_uniform_buffer,
            &self.fxaa_uniform_buffer,
            &self.bloom_down_uniforms,
            &self.bloom_up_uniforms,
            &self.color_sampler,
            &self.point_sampler,
        );
        self.update_bloom_uniforms(queue);
    }

    /// Rewrite the bloom per-level uniforms after a resize (or at creation).
    ///
    /// Each level needs its own source texel size, which is a function of the target dimensions,
    /// so this cannot be a one-off upload.
    fn update_bloom_uniforms(&self, queue: &wgpu::Queue) {
        let (w, h) = self.targets.size;
        queue.write_buffer(
            &self.bloom_down_uniforms[0],
            0,
            bytemuck::bytes_of(&BloomUniform {
                params: [
                    1.0 / w as f32,
                    1.0 / h as f32,
                    self.settings.bloom_threshold,
                    self.settings.bloom_knee,
                ],
                screen: [w as f32, h as f32, 0.0, 0.0],
            }),
        );
        for i in 1..BLOOM_LEVELS {
            let src = &self.targets.bloom[i - 1];
            queue.write_buffer(
                &self.bloom_down_uniforms[i],
                0,
                bytemuck::bytes_of(&BloomUniform {
                    params: [1.0 / src.width as f32, 1.0 / src.height as f32, 0.0, 0.0],
                    screen: [src.width as f32, src.height as f32, i as f32, 0.0],
                }),
            );
        }
        for i in 0..BLOOM_LEVELS.saturating_sub(1) {
            let src = &self.targets.bloom[i + 1];
            queue.write_buffer(
                &self.bloom_up_uniforms[i],
                0,
                bytemuck::bytes_of(&BloomUniform {
                    params: [1.0 / src.width as f32, 1.0 / src.height as f32, 0.0, 0.0],
                    screen: [src.width as f32, src.height as f32, i as f32, 1.0],
                }),
            );
        }
    }

    /// Render one frame.
    ///
    /// Uploads the camera and scene uniforms, culls, then records and submits the whole chain in
    /// one command buffer: shadows, geometry, SSAO, lighting, DoF, transparent, bloom, tonemap and
    /// FXAA. `present` is the swap-chain view to finish into, or `None` in headless mode (the
    /// internal `present` target is used instead). `dt` advances the internal clock, which drives
    /// animated material detail and the dither pattern.
    ///
    /// Returns nothing: pass recording cannot fail in `wgpu`, and a lost surface is the caller's
    /// business (see the viewer's `render_frame` for the reconfigure-on-lost pattern).
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        camera: &Camera,
        present: Option<&wgpu::TextureView>,
        dt: f32,
    ) {
        self.time += dt;
        let aspect = self.targets.size.0 as f32 / self.targets.size.1.max(1) as f32;
        self.update_uniforms(queue, scene, camera, aspect);
        self.update_bloom_uniforms(queue);
        self.cull(camera, aspect);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame-encoder"),
        });

        self.encode_shadows(&mut encoder, scene);
        self.encode_geometry(&mut encoder, scene);
        self.encode_ssao(&mut encoder);
        self.encode_lighting(&mut encoder);
        self.encode_ssr(&mut encoder);
        self.encode_dof(&mut encoder);
        self.sort_transparent(camera);
        self.encode_transparent(&mut encoder, scene);
        self.encode_bloom(&mut encoder);
        self.encode_tonemap(&mut encoder);
        self.encode_fxaa(&mut encoder, present);

        queue.submit(Some(encoder.finish()));
    }

    /// Drop draws whose bounding sphere is outside the view frustum (side planes only: the far
    /// plane is at infinity) and, for shadows, draws too far away to land in any cascade.
    /// Very large objects (terrain) are never culled.
    fn cull(&mut self, camera: &Camera, aspect: f32) {
        let m = camera.proj(aspect) * camera.view();
        let rows = [m.row(0), m.row(1), m.row(2), m.row(3)];
        let norm = |p: glam::Vec4| p / p.truncate().length().max(1e-6);
        let planes = [
            norm(rows[3] + rows[0]),
            norm(rows[3] - rows[0]),
            norm(rows[3] + rows[1]),
            norm(rows[3] - rows[1]),
        ];
        let eye = camera.eye();
        let shadow_reach = self.settings.shadow_distance.min(camera.z_far) + 700.0;
        let in_view = |d: &Draw| -> bool {
            if d.radius > 8000.0 {
                return true;
            }
            let c = glam::Vec3::from(d.center);
            planes
                .iter()
                .all(|p| p.truncate().dot(c) + p.w >= -d.radius)
        };
        self.visible_draws.clear();
        self.visible_draws
            .extend(self.draws.iter().copied().filter(|d| in_view(d)));
        self.visible_shadow_draws.clear();
        self.visible_shadow_draws
            .extend(self.shadow_draws.iter().copied().filter(|d| {
                d.radius > 8000.0
                    || glam::Vec3::from(d.center).distance(eye) - d.radius < shadow_reach
            }));
    }

    /// Write every uniform buffer for this frame: globals, lights, lighting, sky, SSAO, SSR, post,
    /// DoF and FXAA.
    ///
    /// Everything here is a `queue.write_buffer` of a small `Pod` struct, which is why per-frame
    /// CPU cost stays flat as the scene grows. The object buffer is written separately by
    /// [`Renderer::update_objects`] because it only changes when the scene does.
    fn update_uniforms(&self, queue: &wgpu::Queue, scene: &Scene, camera: &Camera, aspect: f32) {
        let view = camera.view();
        let proj = camera.proj(aspect);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();
        let (w, h) = self.targets.size;

        let sun = scene.sun();
        let sun_dir = sun
            .map(|l| l.direction)
            .unwrap_or_else(|| -scene.sky.sun_direction);
        let splits = self.shadow.update(
            queue,
            view,
            camera.fov_y,
            aspect,
            camera.z_near,
            camera.z_far.min(self.settings.shadow_distance),
            sun_dir,
        );

        let mut globals = Globals::default();
        globals.view = view.to_cols_array_2d();
        globals.proj = proj.to_cols_array_2d();
        globals.view_proj = view_proj.to_cols_array_2d();
        globals.inv_view_proj = inv_view_proj.to_cols_array_2d();
        let eye = camera.eye();
        globals.camera_pos = [eye.x, eye.y, eye.z, 1.0];
        globals.screen = [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32];
        globals.near_far = [camera.z_near, camera.z_far, 0.0, 0.0];
        globals.ambient = [
            scene.sky.ground_color.x * self.settings.ambient,
            scene.sky.ground_color.y * self.settings.ambient,
            scene.sky.ground_color.z * self.settings.ambient,
            1.0,
        ];
        globals.sun_dir = [sun_dir.x, sun_dir.y, sun_dir.z, 0.0];
        globals.sun_color = [
            scene.sky.sun_color.x,
            scene.sky.sun_color.y,
            scene.sky.sun_color.z,
            0.0,
        ];
        globals.light_meta = [scene.lights.len().min(MAX_LIGHTS) as u32, 0, 0, 0];
        globals.time = [self.time, dt_or_zero(), 0.0, 0.0];
        globals.shadow_splits = splits;
        globals.shadow_texel = [
            1.0 / self.shadow.size as f32,
            self.shadow.size as f32,
            self.settings.shadow_softness.max(0.0),
            0.0,
        ];
        for (i, m) in self
            .cascade_matrices(queue, view, camera, aspect, sun_dir)
            .iter()
            .enumerate()
        {
            globals.cascade_matrices[i] = m.to_cols_array_2d();
        }
        queue.write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));

        let mut lights: Vec<GpuLight> = Vec::new();
        for light in scene.lights.iter().take(MAX_LIGHTS) {
            let color = light.color * light.intensity;
            let kind = match light.kind {
                LightKind::Directional => 0.0,
                LightKind::Point => 1.0,
                LightKind::Spot => 2.0,
            };
            lights.push(GpuLight {
                position: [light.position.x, light.position.y, light.position.z, kind],
                direction: [
                    light.direction.x,
                    light.direction.y,
                    light.direction.z,
                    light.range,
                ],
                color: [color.x, color.y, color.z, 1.0],
                params: [
                    light.inner_angle,
                    light.outer_angle,
                    0.0,
                    light.enabled as u32 as f32,
                ],
            });
        }
        if lights.is_empty() {
            lights.push(GpuLight {
                position: [0.0; 4],
                direction: [0.0, -1.0, 0.0, 0.0],
                color: [0.0; 4],
                params: [0.0, 0.0, 0.0, 0.0],
            });
        }
        queue.write_buffer(&self.lights_buffer, 0, bytemuck::cast_slice(&lights));

        let lighting = LightingUniform {
            params: [
                self.settings.ssao_enabled as u32 as f32,
                self.settings.shadows_enabled as u32 as f32,
                self.settings.ibl_enabled as u32 as f32,
                self.settings.ibl_intensity,
            ],
            params2: [
                self.settings.shadow_strength,
                (ibl::PREFILTER_MIPS - 1) as f32,
                self.settings.debug_ssao as u32 as f32,
                self.settings.ambient,
            ],
            fog: [
                self.settings.fog_density,
                self.settings.fog_height_falloff,
                self.settings.fog_enabled as u32 as f32,
                self.settings.fog_max_opacity,
            ],
            fog_sun: [
                scene.sky.sun_color.x * self.settings.fog_sun_glow,
                scene.sky.sun_color.y * self.settings.fog_sun_glow,
                scene.sky.sun_color.z * self.settings.fog_sun_glow,
                8.0,
            ],
        };
        queue.write_buffer(
            &self.lighting_uniform_buffer,
            0,
            bytemuck::bytes_of(&lighting),
        );

        let sky = SkyUniform {
            sun_direction: scene.sky.sun_direction.extend(1.0).into(),
            sun_color: (scene.sky.sun_color * scene.sky.sun_intensity)
                .extend(1.0)
                .into(),
            params: [scene.sky.turbidity, scene.sky.exposure, 0.0, 0.0],
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            zenith: scene
                .sky
                .zenith_color
                .extend(scene.sky.horizon_falloff)
                .into(),
            horizon: scene.sky.horizon_color.extend(1.0).into(),
        };
        queue.write_buffer(&self.sky_uniform_buffer, 0, bytemuck::bytes_of(&sky));

        let proj_scale = Vec4::new(proj.x_axis.x, proj.y_axis.y, proj.z_axis.z, proj.w_axis.z);
        let ssao = SsaoUniform {
            proj: proj.to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            inv_view: view.inverse().to_cols_array_2d(),
            params: [
                self.settings.ssao_radius,
                self.settings.ssao_bias,
                self.settings.ssao_intensity,
                1.5,
            ],
            screen: [16.0, w as f32, h as f32, 1.0 / w as f32],
        };
        let _ = proj_scale;
        queue.write_buffer(&self.ssao_uniform_buffer, 0, bytemuck::bytes_of(&ssao));

        let ssr = SsrUniform {
            proj: proj.to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            params: [
                self.settings.ssr_steps,
                0.4,
                0.07,
                if self.settings.ssr_enabled {
                    self.settings.ssr_intensity
                } else {
                    0.0
                },
            ],
            screen: [w as f32, h as f32, self.time, 40.0],
        };
        queue.write_buffer(&self.ssr_uniform_buffer, 0, bytemuck::bytes_of(&ssr));

        let post = PostUniform {
            params: [
                // the camera's own stop (used by the G-LOC grey-out) plus the art-direction
                // stop from `Settings::exposure_ev`
                (2.0_f32).powf(camera.exposure_ev + self.settings.exposure_ev),
                2.2,
                if self.settings.bloom_intensity > 0.0 {
                    self.settings.bloom_intensity
                } else {
                    0.0
                },
                self.settings.vignette,
            ],
            modes: [
                self.settings.tonemap_mode as f32,
                (self.settings.bloom_intensity > 0.0) as u32 as f32,
                self.settings.ssao_enabled as u32 as f32,
                if self.srgb_target { 0.0 } else { 1.0 },
            ],
            screen: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
        };
        queue.write_buffer(&self.post_uniform_buffer, 0, bytemuck::bytes_of(&post));

        let dof = DofUniform {
            inv_proj: proj.inverse().to_cols_array_2d(),
            params: [
                self.settings.dof_focus,
                self.settings.dof_range,
                self.settings.dof_max_radius,
                if self.settings.dof_enabled { 1.0 } else { 0.0 },
            ],
            depth: [camera.z_near, camera.z_far, w as f32, h as f32],
        };
        queue.write_buffer(&self.dof_uniform_buffer, 0, bytemuck::bytes_of(&dof));

        let fxaa = FxaaUniform {
            params: [
                1.0 / w as f32,
                1.0 / h as f32,
                self.settings.fxaa_enabled as u32 as f32,
                0.0,
            ],
        };
        queue.write_buffer(&self.fxaa_uniform_buffer, 0, bytemuck::bytes_of(&fxaa));
    }

    /// Fit the four sun-shadow cascades to the camera's frustum for this frame.
    ///
    /// `shadow::compute_cascades` does the projection-corner fitting; the only thing decided here
    /// is the far distance, which is the camera's far plane clamped by
    /// [`Settings::shadow_distance`] so a 100 km flight view does not waste shadow resolution on
    /// the horizon.
    fn cascade_matrices(
        &self,
        _queue: &wgpu::Queue,
        view: Mat4,
        camera: &Camera,
        aspect: f32,
        sun_dir: Vec3,
    ) -> [Mat4; CASCADE_COUNT] {
        shadow::compute_cascades(
            view,
            camera.fov_y,
            aspect,
            camera.z_near,
            camera.z_far.min(self.settings.shadow_distance),
            sun_dir,
            self.shadow.size,
        )
        .0
    }

    /// Depth-only pass into each cascade slice, with a front-face-culled light-space transform
    /// pushed per layer through a dynamic offset.
    ///
    /// Each cascade clears to 1.0, which is the *near* value under reversed-Z, and only objects
    /// within `shadow_reach` of the eye are submitted (see `cull`).
    fn encode_shadows(&self, encoder: &mut wgpu::CommandEncoder, scene: &Scene) {
        for layer in 0..CASCADE_COUNT as u32 {
            let view = self.shadow.layer_view(layer);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow-pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(
                0,
                &self.shadow.bind_group,
                &[(layer * shadow::SHADOW_STRIDE as u32)],
            );
            pass.set_bind_group(1, &self.objects_bind_group, &[]);
            self.draw_objects(
                &mut pass,
                scene,
                &self.object_index_buffer,
                &self.visible_shadow_draws,
            );
        }
    }

    /// G-buffer pass: albedo, octahedral-ish packed normal, emissive and reversed-Z depth.
    ///
    /// Depth clears to 0.0 (infinity) and compares `GreaterEqual`; the three colour targets clear
    /// to transparent black so an untouched pixel is distinguishable from a black surface.
    fn encode_geometry(&self, encoder: &mut wgpu::CommandEncoder, scene: &Scene) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("geometry-pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.g_albedo.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.g_normal.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.g_emissive.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.geometry_pipeline);
        pass.set_bind_group(0, &self.geometry_globals_bg, &[]);
        pass.set_bind_group(1, &self.objects_bind_group, &[]);
        self.draw_objects(
            &mut pass,
            scene,
            &self.object_index_buffer,
            &self.visible_draws,
        );
    }

    /// Issue one indexed draw per entry, binding the mesh, the material and a 4-byte window of
    /// the slot-index buffer.
    ///
    /// Geometry is not batchable across materials (each material is a separate bind group), but
    /// objects sharing a mesh and material still share their vertex/index buffers, so the CPU cost
    /// per object is a handful of `set_*` calls.
    fn draw_objects<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a Scene,
        index_buffer: &'a wgpu::Buffer,
        list: &'a [Draw],
    ) {
        for draw in list {
            let mesh = &scene.meshes[draw.mesh];
            let material = &scene.materials[draw.material];
            if let Some(bg) = &material.bind_group {
                pass.set_bind_group(2, bg, &[]);
            }
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            let offset = draw.slot as u64 * 4;
            pass.set_vertex_buffer(1, index_buffer.slice(offset..offset + 4));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }

    /// Half-resolution SSAO from depth and normals, then a separable blur to remove the sampling
    /// pattern.
    fn encode_ssao(&self, encoder: &mut wgpu::CommandEncoder) {
        self.fullscreen(
            encoder,
            &self.ssao_pipeline,
            &self.bgs.ssao,
            &self.targets.ssao_raw.view,
            "ssao",
        );
        self.fullscreen(
            encoder,
            &self.ssao_blur_pipeline,
            &self.bgs.ssao_blur,
            &self.targets.ssao_blur.view,
            "ssao-blur",
        );
    }

    /// Full-screen deferred shade into `hdr_a`: the heaviest pass of the frame.
    ///
    /// Reads all three G-buffer targets, the SSAO buffer, the shadow cascades and the IBL cubes,
    /// and produces linear HDR radiance including the sky where depth is still infinite.
    fn encode_lighting(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.hdr_a.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.lighting_pipeline);
        pass.set_bind_group(0, &self.lighting_g0, &[]);
        pass.set_bind_group(1, &self.bgs.lighting_g1, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Screen-space reflection ray march over the lit HDR target, blended back into it.
    ///
    /// Rays start at the reconstructed world position, step through screen space and are rejected
    /// when they leave the frustum or pass behind geometry, so off-screen and behind-camera
    /// reflections simply read as the environment instead of smearing.
    fn encode_ssr(&self, encoder: &mut wgpu::CommandEncoder) {
        self.fullscreen(
            encoder,
            &self.ssr_pipeline,
            &self.bgs.ssr,
            &self.targets.hdr_b.view,
            "ssr",
        );
    }

    /// Depth-of-field gather from the lit HDR target into `hdr_b`, using circle-of-confusion
    /// radius derived from the linear depth and [`Settings::dof_focus`].
    fn encode_dof(&self, encoder: &mut wgpu::CommandEncoder) {
        self.fullscreen(
            encoder,
            &self.dof_pipeline,
            &self.bgs.dof,
            &self.targets.hdr_a.view,
            "dof",
        );
    }

    /// Back-to-front order for the forward transparent pass.
    /// Sort the forward transparency list back to front, measured from the object's origin.
    ///
    /// Per-object sorting is an approximation (two large intersecting panes cannot be ordered
    /// correctly this way), but it is what ordered-dither and blended panels need to look right
    /// and it costs one `sort_by` over a short list.
    fn sort_transparent(&mut self, camera: &Camera) {
        if self.transparent_draws.len() < 2 {
            return;
        }
        let eye = camera.eye();
        self.transparent_draws.sort_by(|a, b| {
            let da = (Vec3::from(a.origin) - eye).length_squared();
            let db = (Vec3::from(b.origin) - eye).length_squared();
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Forward pass for really transparent geometry: alpha blended into the lit HDR target,
    /// depth tested against the deferred depth buffer, no depth writes.
    fn encode_transparent(&self, encoder: &mut wgpu::CommandEncoder, scene: &Scene) {
        if self.transparent_draws.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transparent-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.hdr_a.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.transparent_pipeline);
        pass.set_bind_group(0, &self.lighting_g0, &[]);
        pass.set_bind_group(1, &self.objects_bind_group, &[]);
        self.draw_objects(
            &mut pass,
            scene,
            &self.object_index_buffer,
            &self.transparent_draws,
        );
    }

    /// Bloom: threshold the HDR target into the first bloom mip, downsample to the smallest
    /// level, then upsample additively back up the chain.
    fn encode_bloom(&self, encoder: &mut wgpu::CommandEncoder) {
        self.fullscreen(
            encoder,
            &self.bloom_prefilter,
            &self.bgs.bloom_down[0],
            &self.targets.bloom[0].view,
            "bloom-prefilter",
        );
        for i in 1..BLOOM_LEVELS {
            self.fullscreen(
                encoder,
                &self.bloom_downsample,
                &self.bgs.bloom_down[i],
                &self.targets.bloom[i].view,
                "bloom-down",
            );
        }
        for i in (0..BLOOM_LEVELS.saturating_sub(1)).rev() {
            self.fullscreen(
                encoder,
                &self.bloom_upsample,
                &self.bgs.bloom_up[i],
                &self.targets.bloom[i].view,
                "bloom-up",
            );
        }
    }

    /// Tone-map the HDR target into the LDR target: exposure, bloom mix, curve, vignette and —
    /// when the final surface is not sRGB — gamma encoding.
    fn encode_tonemap(&self, encoder: &mut wgpu::CommandEncoder) {
        self.fullscreen(
            encoder,
            &self.tonemap_pipeline,
            &self.bgs.tonemap,
            &self.targets.ldr.view,
            "tonemap",
        );
    }

    fn encode_fxaa(&self, encoder: &mut wgpu::CommandEncoder, present: Option<&wgpu::TextureView>) {
        let view = match present {
            Some(view) => view,
            None => {
                &self
                    .present
                    .as_ref()
                    .expect("headless renderer has a present target")
                    .view
            }
        };
        self.fullscreen(encoder, &self.fxaa_pipeline, &self.bgs.fxaa, view, "fxaa");
    }

    /// Run a full-screen triangle through `pipeline` into `target`.
    ///
    /// The vertex shader generates the triangle from `vertex_index` alone, so no vertex buffer is
    /// bound; every post pass goes through this one helper, which is why they all share a shape.
    fn fullscreen(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        bind_group: &wgpu::BindGroup,
        target: &wgpu::TextureView,
        label: &str,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Placeholder for the per-frame delta stored in `Globals::time.y`.
///
/// Nothing reads that component yet; the clock proper is `Globals::time.x`. Kept as a named
/// function rather than a literal so the slot has an obvious owner when something does read it.
fn dt_or_zero() -> f32 {
    0.0
}

/// Additive blending, used for the bloom upsample chain and the sky disc.
fn additive_blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// Bind the object storage buffer (transforms, material parameters, flags) for a draw pass.
fn create_objects_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("objects-bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

/// A texture entry for a bind group layout: fragment-stage, filtered, 2D, not multisampled.
fn tex_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// A uniform-buffer entry with an explicit minimum binding size.
///
/// The size is passed explicitly because `wgpu` validates it on every bind: a layout that declares
/// less than the shader reads is a validation error at draw time, not at creation time.
fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    size: u64,
    dynamic: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

/// A sampler entry with the sampling kind the shader declares (filtering vs comparison).
fn sampler_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
}

/// The sample type of every colour, depth and data texture in the pipeline.
fn float() -> wgpu::TextureSampleType {
    wgpu::TextureSampleType::Float { filterable: true }
}

/// SSAO needs the depth buffer, the normal buffer, its uniform block and a point sampler.
fn create_ssao_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ssao-bgl"),
        entries: &[
            uniform_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                std::mem::size_of::<SsaoUniform>() as u64,
                false,
            ),
            tex_entry(
                1,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Depth,
                wgpu::TextureViewDimension::D2,
            ),
            tex_entry(
                2,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            sampler_entry(
                3,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::NonFiltering,
            ),
            tex_entry(
                4,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
        ],
    })
}

/// Group 0 of the lighting pass: globals, the light list, the shadow map at each cascade slice,
/// the IBL cubes and the lighting uniforms.
fn create_lighting_layout_g0(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lighting-g0-bgl"),
        entries: &[
            uniform_entry(
                0,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                std::mem::size_of::<Globals>() as u64,
                false,
            ),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuLight>() as u64),
                },
                count: None,
            },
            tex_entry(
                2,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                wgpu::TextureSampleType::Depth,
                wgpu::TextureViewDimension::D2Array,
            ),
            sampler_entry(
                3,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                wgpu::SamplerBindingType::Comparison,
            ),
            tex_entry(
                4,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                float(),
                wgpu::TextureViewDimension::Cube,
            ),
            tex_entry(
                5,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                float(),
                wgpu::TextureViewDimension::Cube,
            ),
            tex_entry(
                6,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            sampler_entry(
                7,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            uniform_entry(
                8,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                std::mem::size_of::<LightingUniform>() as u64,
                false,
            ),
            tex_entry(
                9,
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                float(),
                wgpu::TextureViewDimension::Cube,
            ),
            sampler_entry(
                10,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::NonFiltering,
            ),
        ],
    })
}

/// Group 1 of the lighting pass: the G-buffer targets, SSAO, SSR, bloom and their samplers.
fn create_lighting_layout_g1(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lighting-g1-bgl"),
        entries: &[
            tex_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            tex_entry(
                1,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            tex_entry(
                2,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            tex_entry(
                3,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Depth,
                wgpu::TextureViewDimension::D2,
            ),
            sampler_entry(
                4,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
            tex_entry(
                5,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
        ],
    })
}

/// SSR reads the lit colour, the depth buffer and its own uniform block.
fn create_ssr_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ssr-bgl"),
        entries: &[
            uniform_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                std::mem::size_of::<SsrUniform>() as u64,
                false,
            ),
            tex_entry(
                1,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            tex_entry(
                2,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Depth,
                wgpu::TextureViewDimension::D2,
            ),
            tex_entry(
                3,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            sampler_entry(
                4,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
        ],
    })
}

/// Shared by every bloom step: one source texture, one uniform block, one sampler.
fn create_bloom_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bloom-bgl"),
        entries: &[
            uniform_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                std::mem::size_of::<BloomUniform>() as u64,
                false,
            ),
            tex_entry(
                1,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            sampler_entry(
                2,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
        ],
    })
}

/// DoF reads the lit colour and the depth buffer, plus its uniform block.
fn create_dof_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dof-bgl"),
        entries: &[
            uniform_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                std::mem::size_of::<DofUniform>() as u64,
                false,
            ),
            tex_entry(
                1,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            tex_entry(
                2,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Depth,
                wgpu::TextureViewDimension::D2,
            ),
            sampler_entry(
                3,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
        ],
    })
}

/// Tonemap reads the blended HDR target (scene + bloom) and its uniform block.
fn create_tonemap_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tonemap-bgl"),
        entries: &[
            uniform_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                std::mem::size_of::<PostUniform>() as u64,
                false,
            ),
            tex_entry(
                1,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            tex_entry(
                2,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            sampler_entry(
                3,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
        ],
    })
}

/// FXAA reads the LDR target and its uniform block.
fn create_fxaa_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fxaa-bgl"),
        entries: &[
            uniform_entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                std::mem::size_of::<FxaaUniform>() as u64,
                false,
            ),
            tex_entry(
                1,
                wgpu::ShaderStages::FRAGMENT,
                float(),
                wgpu::TextureViewDimension::D2,
            ),
            sampler_entry(
                2,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::SamplerBindingType::Filtering,
            ),
        ],
    })
}

/// The G-buffer pipeline: per-vertex attributes plus the instance slot, three colour targets and
/// reversed-Z depth. Blends ordered-dither coverage rather than real alpha.
fn create_geometry_pipeline(
    device: &wgpu::Device,
    globals_bgl: &wgpu::BindGroupLayout,
    objects_bgl: &wgpu::BindGroupLayout,
    material_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = shader::create(
        device,
        "geometry.wgsl",
        include_str!("../../shaders/geometry.wgsl"),
    );
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("geometry-pl"),
        bind_group_layouts: &[globals_bgl, objects_bgl, material_bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("geometry-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            compilation_options: Default::default(),
            buffers: &[
                crate::mesh::Vertex::layout(),
                crate::mesh::Vertex::instance_layout(),
            ],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview: None,
        cache: None,
    })
}

/// The forward transparency pipeline: alpha blended, depth tested but not written, lit by the
/// same lighting code as the deferred pass.
fn create_transparent_pipeline(
    device: &wgpu::Device,
    lighting_bgl: &wgpu::BindGroupLayout,
    objects_bgl: &wgpu::BindGroupLayout,
    material_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = shader::create(
        device,
        "transparent.wgsl",
        include_str!("../../shaders/transparent.wgsl"),
    );
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("transparent-pl"),
        bind_group_layouts: &[lighting_bgl, objects_bgl, material_bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("transparent-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            compilation_options: Default::default(),
            buffers: &[
                crate::mesh::Vertex::layout(),
                crate::mesh::Vertex::instance_layout(),
            ],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::GreaterEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

/// Depth-only pipeline for the shadow cascades: no colour target, front-face culling so thin
/// geometry contributes to the shadow volume from both sides.
fn create_shadow_pipeline(
    device: &wgpu::Device,
    shadow_bgl: &wgpu::BindGroupLayout,
    objects_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = shader::create(
        device,
        "shadow.wgsl",
        include_str!("../../shaders/shadow.wgsl"),
    );
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shadow-pl"),
        bind_group_layouts: &[shadow_bgl, objects_bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            compilation_options: Default::default(),
            buffers: &[
                crate::mesh::Vertex::layout(),
                crate::mesh::Vertex::instance_layout(),
            ],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState {
                constant: 1,
                slope_scale: 2.0,
                clamp: 0.0,
            },
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: None,
        multiview: None,
        cache: None,
    })
}

/// A full-screen post pass with a single bind group, built from WGSL source text.
fn create_screen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &str,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let shader = shader::create(device, fragment_entry, source);
    create_screen_pipeline_with_layout(device, layout, &shader, fragment_entry, format, blend)
}

/// A full-screen post pass with several bind groups (the lighting pass needs two).
fn create_screen_pipeline_multi(
    device: &wgpu::Device,
    layouts: &[&wgpu::BindGroupLayout],
    source: &str,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let shader = shader::create(device, fragment_entry, source);
    create_screen_pipeline_with_layouts(device, layouts, &shader, fragment_entry, format, blend)
}

/// [`create_screen_pipeline_multi`] for a shader module that is already compiled, so several
/// passes can share one module.
fn create_screen_pipeline_with_layouts(
    device: &wgpu::Device,
    layouts: &[&wgpu::BindGroupLayout],
    shader: &wgpu::ShaderModule,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(fragment_entry),
        bind_group_layouts: layouts,
        push_constant_ranges: &[],
    });
    create_screen_pipeline_inner(device, &pl_layout, shader, fragment_entry, format, blend)
}

/// [`create_screen_pipeline`] for an already-compiled shader module.
fn create_screen_pipeline_with_layout(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(fragment_entry),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    create_screen_pipeline_inner(device, &pl_layout, shader, fragment_entry, format, blend)
}

/// The one place a full-screen pipeline is actually described: no vertex buffers, no culling,
/// no depth, triangle list, and a `vs_fullscreen` vertex entry point.
fn create_screen_pipeline_inner(
    device: &wgpu::Device,
    pl_layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment_entry),
        layout: Some(&pl_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_fullscreen",
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: fragment_entry,
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

#[allow(clippy::too_many_arguments)]
/// Build the per-target bind groups used by the post passes after a resize.
///
/// Every group here depends on a texture that `resize` recreated, which is why they cannot be
/// built once and kept.
fn build_target_bindings(
    device: &wgpu::Device,
    targets: &Targets,
    ssao_uniform_buffer: &wgpu::Buffer,
    ssr_uniform_buffer: &wgpu::Buffer,
    dof_uniform_buffer: &wgpu::Buffer,
    post_uniform_buffer: &wgpu::Buffer,
    fxaa_uniform_buffer: &wgpu::Buffer,
    bloom_down_uniforms: &[wgpu::Buffer],
    bloom_up_uniforms: &[wgpu::Buffer],
    color_sampler: &wgpu::Sampler,
    point_sampler: &wgpu::Sampler,
) -> TargetBindGroups {
    let ssao_layout = create_ssao_layout(device);
    let ssao = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssao-bg"),
        layout: &ssao_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ssao_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.depth.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&targets.g_normal.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(point_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&targets.ssao_blur.view),
            },
        ],
    });
    let ssao_blur = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssao-blur-bg"),
        layout: &ssao_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ssao_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.depth.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&targets.g_normal.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(point_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&targets.ssao_raw.view),
            },
        ],
    });

    let lighting_g1_layout = create_lighting_layout_g1(device);
    let lighting_g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lighting-g1"),
        layout: &lighting_g1_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&targets.g_albedo.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.g_normal.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&targets.g_emissive.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&targets.depth.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(point_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&targets.ssao_blur.view),
            },
        ],
    });

    let ssr_layout = create_ssr_layout(device);
    let ssr = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssr-bg"),
        layout: &ssr_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ssr_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.hdr_a.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&targets.depth.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&targets.g_normal.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(color_sampler),
            },
        ],
    });

    let dof_layout = create_dof_layout(device);
    let dof = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dof-bg"),
        layout: &dof_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: dof_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.hdr_b.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&targets.depth.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(color_sampler),
            },
        ],
    });

    let bloom_layout = create_bloom_layout(device);
    let mut bloom_down = Vec::new();
    for i in 0..BLOOM_LEVELS {
        let src = if i == 0 {
            &targets.hdr_a.view
        } else {
            &targets.bloom[i - 1].view
        };
        bloom_down.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-down-bg"),
            layout: &bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_down_uniforms[i].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(color_sampler),
                },
            ],
        }));
    }
    let mut bloom_up = Vec::new();
    for i in 0..BLOOM_LEVELS.saturating_sub(1) {
        let src = &targets.bloom[i + 1].view;
        bloom_up.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-up-bg"),
            layout: &bloom_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bloom_up_uniforms[i].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(color_sampler),
                },
            ],
        }));
    }

    let tonemap_layout = create_tonemap_layout(device);
    let tonemap = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tonemap-bg"),
        layout: &tonemap_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: post_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.hdr_a.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&targets.bloom[0].view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(color_sampler),
            },
        ],
    });

    let fxaa_layout = create_fxaa_layout(device);
    let fxaa = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fxaa-bg"),
        layout: &fxaa_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: fxaa_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&targets.ldr.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(color_sampler),
            },
        ],
    });

    TargetBindGroups {
        ssao,
        ssao_blur,
        lighting_g1,
        ssr,
        dof,
        bloom_down,
        bloom_up,
        tonemap,
        fxaa,
    }
}

#[allow(clippy::too_many_arguments)]
/// Build the lighting pass's group 0, rebinding the shadow map and IBL cubes.
///
/// Rebuilding is needed after [`Renderer::refresh_environment`], because the IBL textures it
/// produces are new objects even though their handles look the same.
fn build_lighting_g0(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    globals_buffer: &wgpu::Buffer,
    lights_buffer: &wgpu::Buffer,
    shadow: &Shadow,
    ibl: &Ibl,
    lighting_uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lighting-g0"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lights_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&shadow.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&shadow.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&ibl.irradiance_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&ibl.prefilter_full_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&ibl.brdf_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(&ibl.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: lighting_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&ibl.env_view),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::Sampler(&shadow.depth_sampler),
            },
        ],
    })
}
