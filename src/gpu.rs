//! Thin helpers over `wgpu` resource creation.
//!
//! Nothing here is clever: the point is that every buffer, texture and view in the renderer is
//! created in one of these functions, so the size/usage/label conventions live in one file and a
//! new pipeline cannot quietly introduce a different one. Several helpers clamp their inputs
//! (`size.max(16)`, `width.max(1)`, `mip_levels.max(1)`) so a zero-sized scene or a 0×0 window
//! degrades into a valid resource instead of a validation error.

use bytemuck::Pod;
use wgpu::util::DeviceExt;

/// Create a buffer pre-filled from a slice of `Pod` data.
///
/// `usage` is taken verbatim, so callers must already include the bind usage they intend; the
/// data is written through the staging path `create_buffer_init` uses.
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

/// Create an uninitialised buffer for per-frame uploads, e.g. the light and object lists.
///
/// The size is clamped to 16 bytes because `wgpu` rejects zero-sized buffers, and an empty scene
/// is a normal state at start-up. Use [`uniform_buffer`] for anything bound as a uniform.
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

/// Create a uniform buffer holding a single `Pod` value, with `COPY_DST` for later writes.
///
/// The bind group layout's min binding size must match `size_of::<T>()`; the renderer builds both
/// from the same type so this holds by construction.
pub fn uniform_buffer<T: Pod>(device: &wgpu::Device, label: &str, data: &T) -> wgpu::Buffer {
    buffer_init(
        device,
        label,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        std::slice::from_ref(data),
    )
}

/// Create a 2D texture with explicit format, usage, mip count and sample count.
///
/// Used for every render target and every uploaded image; the renderer passes
/// `sample_count: 1` everywhere, so the parameter exists for future MSAA surfaces rather than for
/// current use.
#[allow(clippy::too_many_arguments)]
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

/// Create a cube texture: a D2 texture with six array layers, viewed as a cube.
///
/// This is the IBL storage shape — the sky is rendered into it and the prefiltered radiance chain
/// is stored in its mips.
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

/// Create a 2D array texture with the given layer count, e.g. one layer per shadow cascade.
#[allow(clippy::too_many_arguments)]
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

/// The default view: every mip level, every layer, the texture's own dimension.
///
/// Used for material textures and for render targets that are sampled whole.
pub fn default_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// A cube view of one mip level, for binding a specific roughness level of the radiance chain.
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

/// A 2D view of a single cube face at one mip level, so a face can be rendered into as a target.
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

/// A 2D view of one array layer (one shadow cascade), for rendering into or sampling.
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

/// Write a rectangle of pixels into one mip level of one array layer.
///
/// `bytes_per_row` is explicit because `queue.write_texture` requires it for multi-row images and
/// callers here always have tightly packed RGBA8 rows; `layer` selects the array slice (0 for
/// plain 2D textures).
#[allow(clippy::too_many_arguments)]
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
///
/// Each level is the previous one halved with a triangle filter, continuing until 1×1, which is
/// what uploads need: `wgpu` has no CPU-side mip generation, and doing it in a compute pass would
/// add a pipeline per format for a one-off cost at load time.
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

/// Round `value` up to the next multiple of `alignment`.
///
/// Needed for texture-to-buffer copies: `wgpu` requires `bytes_per_row` to be a multiple of
/// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`] (256), which is why a 1280-wide RGBA8 readback has
/// three rows' worth of padding to strip afterwards.
pub fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

/// The device limits to request: WebGPU defaults, lowered where the adapter cannot reach them.
///
/// Starting from [`wgpu::Limits::default`] (rather than the adapter's maxima) keeps the crate
/// working identically on every backend and prevents an over-optimistic request from failing on a
/// weaker device. The fields that matter here are texture size (render targets, shadow map), bind
/// groups, sampled textures per stage (five material maps plus the G-buffer and IBL) and the
/// storage/uniform binding sizes for the light and object buffers.
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
