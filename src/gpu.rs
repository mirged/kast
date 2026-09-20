use bytemuck::Pod;
use wgpu::util::DeviceExt;

pub fn buffer_init<T: Pod>(
    device: &wgpu::Device,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

pub fn buffer_empty(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage,
        mapped_at_creation: false,
    })
}

pub fn uniform_buffer<T: Pod>(device: &wgpu::Device, label: &str, data: &T) -> wgpu::Buffer {
    buffer_init(
        device,
        label,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        std::slice::from_ref(data),
    )
}

pub fn texture_2d(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    mip_levels: u32,
    sample_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_levels.max(1),
        sample_count: sample_count.max(1),
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

pub fn texture_cube(
    device: &wgpu::Device,
    label: &str,
    size: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    mip_levels: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.max(1),
            height: size.max(1),
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels.max(1),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

pub fn texture_2d_array(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    layers: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    mip_levels: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: layers,
        },
        mip_level_count: mip_levels.max(1),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

pub fn default_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub fn cube_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("cube-view"),
        format: None,
        dimension: Some(wgpu::TextureViewDimension::Cube),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: None,
    })
}

pub fn cube_face_view(texture: &wgpu::Texture, mip: u32, face: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("cube-face-view"),
        format: None,
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: face,
        array_layer_count: Some(1),
    })
}

pub fn array_layer_view(texture: &wgpu::Texture, layer: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("array-layer-view"),
        format: None,
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: layer,
        array_layer_count: Some(1),
    })
}

pub fn upload_2d(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer: u32,
    mip: u32,
    width: u32,
    height: u32,
    data: &[u8],
    bytes_per_row: u32,
) {
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: mip,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Build a full RGBA8 mip pyramid on the CPU using box-ish filtering.
pub fn generate_mips_rgba8(base: &image::RgbaImage) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = Vec::new();
    let mut w = base.width().max(1);
    let mut h = base.height().max(1);
    levels.push((w, h, base.as_raw().clone()));
    let mut current = base.clone();
    while w > 1 || h > 1 {
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        let resized =
            image::imageops::resize(&current, nw, nh, image::imageops::FilterType::Triangle);
        levels.push((nw, nh, resized.as_raw().clone()));
        current = resized;
        w = nw;
        h = nh;
    }
    levels
}

pub fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

pub fn gpu_limits_for(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let mut limits = wgpu::Limits::default();
    let supported = adapter.limits();
    limits.max_texture_dimension_2d = limits
        .max_texture_dimension_2d
        .min(supported.max_texture_dimension_2d);
    limits.max_bind_groups = limits.max_bind_groups.min(supported.max_bind_groups);
    limits.max_sampled_textures_per_shader_stage = limits
        .max_sampled_textures_per_shader_stage
        .min(supported.max_sampled_textures_per_shader_stage);
    limits.max_storage_buffers_per_shader_stage = limits
        .max_storage_buffers_per_shader_stage
        .min(supported.max_storage_buffers_per_shader_stage);
    limits.max_storage_buffer_binding_size = limits
        .max_storage_buffer_binding_size
        .min(supported.max_storage_buffer_binding_size);
    limits.max_uniform_buffer_binding_size = limits
        .max_uniform_buffer_binding_size
        .min(supported.max_uniform_buffer_binding_size);
    limits
}
