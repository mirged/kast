//! Image-based lighting: bake the analytical sky into cubemaps the shading pass can sample.
//!
//! Four products come out of one command buffer, all generated on the GPU at start-up (and again
//! whenever [`Renderer::refresh_environment`](crate::renderer::Renderer::refresh_environment) is
//! called):
//!
//! - **environment cube** — the sky rendered into a 256² cube, the source for the passes below,
//! - **irradiance cube** — cosine-convolved diffuse ambient, 32² per face,
//! - **prefiltered radiance cube** — GGX-prefiltered specular reflections, one mip per roughness
//!   level, 128² down to 4²,
//! - **BRDF LUT** — the split-sum integration term (`Rg16Float`), computed once for all
//!   materials.
//!
//! Splitting diffuse and specular this way is what makes a material's roughness actually change how
//! sharp its reflections are, without tracing a single ray.
//!
//! All four passes use one pipeline layout and one uniform buffer, with the per-pass parameters
//! selected through a dynamic offset — the buffer is pre-filled with 49 slots (6 env faces, 6
//! irradiance faces, 36 prefilter mip/face pairs, 1 BRDF), so the whole bake is a burst of
//! full-screen draws with no buffer writes in between.

use crate::gpu;
use crate::types::{IblUniform, SkyUniform};

/// Edge length of the source environment cube, in texels per face.
///
/// This is the resolution the sky itself is drawn at; the specular chain is derived from it, so it
/// bounds how crisp a mirror reflection can be.
pub const ENV_SIZE: u32 = 256;

/// Edge length of one irradiance-cube face. Diffuse ambient is extremely low frequency, so 32² is
/// more than enough and keeps the convolution cheap.
pub const IRRADIANCE_SIZE: u32 = 32;

/// Edge length of mip 0 of the prefiltered radiance cube.
pub const PREFILTER_SIZE: u32 = 128;

/// Number of roughness levels in the prefiltered radiance cube; mip `i` corresponds to roughness
/// `i / (PREFILTER_MIPS - 1)`, so the last mip is a fully rough lobe.
pub const PREFILTER_MIPS: u32 = 6;

/// Edge length of the BRDF integration LUT.
pub const BRDF_SIZE: u32 = 256;

/// Stride between dynamic-offset uniform slots in the IBL parameter buffer.
///
/// 256 bytes is the alignment `wgpu` requires for a dynamic offset, which is why the buffer is
/// padded far beyond the 32 bytes an [`IblUniform`] actually occupies.
const DYNAMIC_STRIDE: u64 = 256;

/// The baked environment: source cube, irradiance cube, prefiltered radiance chain and BRDF LUT.
///
/// Owned by the renderer and re-created (not updated in place) when the sky changes, because the
/// prefilter chain's mip count and views are fixed at allocation time.
#[allow(dead_code)]
pub struct Ibl {
    /// The sky itself, rendered into a cube. Useful as a background and as the prefilter source.
    pub env_texture: wgpu::Texture,
    /// Cube view of the environment, every face at mip 0.
    pub env_view: wgpu::TextureView,
    /// Cosine-convolved diffuse ambient, `Rgba16Float`.
    pub irradiance_texture: wgpu::Texture,
    /// Cube view of the irradiance cube.
    pub irradiance_view: wgpu::TextureView,
    /// GGX-prefiltered specular reflections with one mip per roughness level.
    pub prefilter_texture: wgpu::Texture,
    /// One cube view per mip level, so a shader can sample a specific roughness directly.
    pub prefilter_views: Vec<wgpu::TextureView>,
    /// Cube view spanning every mip, for `textureSampleLevel` with an explicit LOD.
    pub prefilter_full_view: wgpu::TextureView,
    /// The split-sum BRDF integration term, `Rg16Float` (scale in R, bias in G).
    pub brdf_texture: wgpu::Texture,
    /// 2D view of the BRDF LUT.
    pub brdf_view: wgpu::TextureView,
    /// Trilinear sampler clamped to the mip chain, shared by all four products.
    pub sampler: wgpu::Sampler,
    /// Layout of the groups created here, reused by the lighting pass.
    pub layout: wgpu::BindGroupLayout,
}

impl Ibl {
    /// Bake all four products for `sky` and return them ready to bind.
    ///
    /// This compiles four pipelines and records 49 full-screen draws into a single command buffer,
    /// so it is a start-up cost: a few milliseconds on a discrete GPU, noticeably more on a
    /// software rasterizer. Call it when the sky changes, not per frame.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, sky: &SkyUniform) -> Self {
        let rt = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        let env_texture = gpu::texture_cube(
            device,
            "env-cube",
            ENV_SIZE,
            wgpu::TextureFormat::Rgba16Float,
            rt,
            1,
        );
        let env_view = gpu::cube_view(&env_texture, 0);
        let irradiance_texture = gpu::texture_cube(
            device,
            "irradiance-cube",
            IRRADIANCE_SIZE,
            wgpu::TextureFormat::Rgba16Float,
            rt,
            1,
        );
        let irradiance_view = gpu::cube_view(&irradiance_texture, 0);
        let prefilter_texture = gpu::texture_cube(
            device,
            "prefilter-cube",
            PREFILTER_SIZE,
            wgpu::TextureFormat::Rgba16Float,
            rt,
            PREFILTER_MIPS,
        );
        let prefilter_full_view = gpu::cube_view(&prefilter_texture, 0);
        let prefilter_views = (0..PREFILTER_MIPS)
            .map(|m| gpu::cube_view(&prefilter_texture, m))
            .collect();
        let brdf_texture = gpu::texture_2d(
            device,
            "brdf-lut",
            BRDF_SIZE,
            BRDF_SIZE,
            wgpu::TextureFormat::Rg16Float,
            rt,
            1,
            1,
        );
        let brdf_view = gpu::default_view(&brdf_texture);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: PREFILTER_MIPS as f32,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let layout = create_ibl_layout(device);

        let shader = crate::renderer::shader::create(
            device,
            "ibl.wgsl",
            include_str!("../../shaders/ibl.wgsl"),
        );

        // One shader module, four entry points: the sky draw, the cosine convolution, the GGX
        // prefilter and the BRDF integration.
        let pipeline_env = create_fullscreen_pipeline(
            device,
            &layout,
            &shader,
            "fs_env",
            wgpu::TextureFormat::Rgba16Float,
        );
        let pipeline_irradiance = create_fullscreen_pipeline(
            device,
            &layout,
            &shader,
            "fs_irradiance",
            wgpu::TextureFormat::Rgba16Float,
        );
        let pipeline_prefilter = create_fullscreen_pipeline(
            device,
            &layout,
            &shader,
            "fs_prefilter",
            wgpu::TextureFormat::Rgba16Float,
        );
        let pipeline_brdf = create_fullscreen_pipeline(
            device,
            &layout,
            &shader,
            "fs_brdf",
            wgpu::TextureFormat::Rg16Float,
        );

        // Slot map, by construction: 0..6 environment faces, 6..12 irradiance faces,
        // 12..48 prefilter (face + 6 * mip), 48 the BRDF LUT.
        let mut uniforms = vec![
            IblUniform {
                params: [0.0; 4],
                face: [0; 4]
            };
            64
        ];
        let mut sky = *sky;
        // The face index and mode are per-draw, not per-sky; they are carried by IblUniform.
        sky.params = [0.0; 4];

        let mut put = |slot: usize, u: IblUniform| {
            uniforms[slot] = u;
        };

        for face in 0..6u32 {
            put(
                face as usize,
                IblUniform {
                    params: [0.0, 0.0, 0.0, ENV_SIZE as f32],
                    face: [face, 1, 0, 0],
                },
            );
        }
        for face in 0..6u32 {
            put(
                6 + face as usize,
                IblUniform {
                    params: [0.0, 0.0, 16.0, IRRADIANCE_SIZE as f32],
                    face: [face, 1, 0, 0],
                },
            );
        }
        for mip in 0..PREFILTER_MIPS {
            for face in 0..6u32 {
                let midx = face as usize + mip as usize * 6;
                uniforms[12 + midx] = IblUniform {
                    params: [
                        mip as f32 / (PREFILTER_MIPS - 1) as f32,
                        mip as f32,
                        48.0,
                        PREFILTER_SIZE as f32,
                    ],
                    face: [face, PREFILTER_MIPS, 0, 0],
                };
            }
        }
        uniforms[48] = IblUniform {
            params: [0.0; 4],
            face: [0, 1, 0, 0],
        };

        // Pack the slots into one buffer at the dynamic-offset alignment the layout declares.
        let slot_bytes = DYNAMIC_STRIDE as usize;
        let mut uniform_data = vec![0u8; uniforms.len() * slot_bytes];
        for (i, uniform) in uniforms.iter().enumerate() {
            let bytes = bytemuck::bytes_of(uniform);
            uniform_data[i * slot_bytes..i * slot_bytes + bytes.len()].copy_from_slice(bytes);
        }
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ibl-uniforms"),
            size: uniform_data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, &uniform_data);

        let sky_buffer = gpu::uniform_buffer(device, "ibl-sky", &sky);

        let make_bg = |label: &str, source: &wgpu::TextureView, uniform_buffer: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sky_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: uniform_buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<IblUniform>() as u64),
                        }),
                    },
                ],
            })
        };

        // Source cube for the irradiance/prefilter/brdf passes.
        let bind_group = make_bg("ibl-bg", &env_view, &uniform_buffer);
        // The environment passes render into the environment cube itself, so they
        // must not bind it as a sampled source. `fs_env` does not read the cube.
        let env_bind_group = make_bg("ibl-env-bg", &irradiance_view, &uniform_buffer);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl-encoder"),
        });

        for face in 0..6u32 {
            let view = gpu::cube_face_view(&env_texture, 0, face);
            run_fullscreen(
                &mut encoder,
                &pipeline_env,
                &env_bind_group,
                face as u64 * DYNAMIC_STRIDE,
                &view,
                "env-face",
            );
        }
        for face in 0..6u32 {
            let view = gpu::cube_face_view(&irradiance_texture, 0, face);
            run_fullscreen(
                &mut encoder,
                &pipeline_irradiance,
                &bind_group,
                (6 + face) as u64 * DYNAMIC_STRIDE,
                &view,
                "irradiance-face",
            );
        }
        for mip in 0..PREFILTER_MIPS {
            for face in 0..6u32 {
                let view = gpu::cube_face_view(&prefilter_texture, mip, face);
                let slot = 12 + face as usize + mip as usize * 6;
                run_fullscreen(
                    &mut encoder,
                    &pipeline_prefilter,
                    &bind_group,
                    slot as u64 * DYNAMIC_STRIDE,
                    &view,
                    "prefilter-face",
                );
            }
        }
        run_fullscreen(
            &mut encoder,
            &pipeline_brdf,
            &bind_group,
            48 * DYNAMIC_STRIDE,
            &brdf_view,
            "brdf-lut",
        );

        queue.submit(Some(encoder.finish()));

        Self {
            env_texture,
            env_view,
            irradiance_texture,
            irradiance_view,
            prefilter_texture,
            prefilter_views,
            prefilter_full_view,
            brdf_texture,
            brdf_view,
            sampler,
            layout,
        }
    }
}

/// Layout shared by every IBL pass: the sky parameters, the source cube, a sampler, and the
/// per-draw parameters at a dynamic offset.
///
/// Slot 3 is dynamic because all 49 draws share one buffer; the offset selects the face, mip,
/// sample count and roughness for that draw.
pub fn create_ibl_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ibl-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<SkyUniform>() as u64
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<IblUniform>() as u64
                    ),
                },
                count: None,
            },
        ],
    })
}

/// Build a no-depth, no-blend full-screen pipeline with one bind group, writing `format`.
///
/// `fragment_entry` picks which of the four IBL passes this pipeline runs; the vertex stage is
/// always the shared `vs_fullscreen`.
pub fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ibl-pl"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
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
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

/// Draw one full-screen triangle into `target` with the given pipeline and dynamic offset.
///
/// Used 49 times during the bake; each call is its own render pass because the target view differs.
#[allow(clippy::too_many_arguments)]
pub fn run_fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    dynamic_offset: u64,
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
    pass.set_bind_group(0, bind_group, &[dynamic_offset as u32]);
    pass.draw(0..3, 0..1);
}
