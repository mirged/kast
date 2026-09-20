use crate::gpu;
use crate::types::{IblUniform, SkyUniform};

pub const ENV_SIZE: u32 = 256;
pub const IRRADIANCE_SIZE: u32 = 32;
pub const PREFILTER_SIZE: u32 = 128;
pub const PREFILTER_MIPS: u32 = 6;
pub const BRDF_SIZE: u32 = 256;
const DYNAMIC_STRIDE: u64 = 256;

#[allow(dead_code)]
pub struct Ibl {
    pub env_texture: wgpu::Texture,
    pub env_view: wgpu::TextureView,
    pub irradiance_texture: wgpu::Texture,
    pub irradiance_view: wgpu::TextureView,
    pub prefilter_texture: wgpu::Texture,
    pub prefilter_views: Vec<wgpu::TextureView>,
    pub prefilter_full_view: wgpu::TextureView,
    pub brdf_texture: wgpu::Texture,
    pub brdf_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub layout: wgpu::BindGroupLayout,
}

impl Ibl {
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

        let mut uniforms = vec![
            IblUniform {
                params: [0.0; 4],
                face: [0; 4]
            };
            64
        ];
        let mut sky = *sky;
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
