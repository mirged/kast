//! The vertex format and the GPU-side mesh once geometry has been uploaded.
//!
//! A [`Mesh`] owns one vertex buffer and one index buffer and is treated as static for its whole
//! lifetime; the scene refers to it by index. Objects that move, disappear or change material are
//! expressed through per-object data (see [`crate::scene::SceneObject`]), never by touching these
//! buffers.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// One vertex: position, normal, tangent frame and UV, packed for direct upload.
///
/// `#[repr(C)]` plus [`Pod`] is what makes the struct safe to hand to the GPU as raw bytes, so the
/// field order and types here must match what the shaders declare in their `VertexInput` — see
/// [`Vertex::layout`], which is the single place that mapping is written down.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Vertex {
    /// Object-space position, in metres.
    pub position: [f32; 3],
    /// Object-space unit normal.
    pub normal: [f32; 3],
    /// Tangent frame: `xyz` is the unit tangent, `w` is the bitangent handedness (±1). A shader
    /// reconstructs the bitangent as `cross(normal, tangent.xyz) * tangent.w`.
    pub tangent: [f32; 4],
    /// Texture coordinates, 0..1 across the authored unwrap.
    pub uv: [f32; 2],
}

impl Vertex {
    /// The per-vertex buffer layout: shader locations 0..3 for position, normal, tangent and UV.
    ///
    /// `const` so pipelines can be built without allocating; the attribute table is a
    /// compile-time constant because `wgpu::VertexBufferLayout` borrows it for the pipeline's
    /// lifetime.
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x4,
            3 => Float32x2,
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRS,
        }
    }

    /// The per-instance buffer layout: shader location 4, one `u32` slot per instance.
    ///
    /// The slot is an index into the renderer's object buffer, so a single draw can cover every
    /// object that shares a mesh and a material; the geometry shader reads the transform and
    /// material from the object buffer instead of from per-instance matrices.
    pub const fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
        const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![4 => Uint32];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRS,
        }
    }
}

/// An uploaded mesh: vertex and index buffers plus its object-space bounding box.
///
/// Created by [`Mesh::new`] and owned by [`crate::scene::Scene::meshes`]; objects point at it by
/// index. The bounds are what the renderer's frustum and shadow-range culling test against, so
/// they are computed from the vertices rather than taken on trust.
pub struct Mesh {
    /// Debug name, also used as the buffer labels and in log output.
    #[allow(dead_code)]
    pub name: String,
    /// Interleaved [`Vertex`] data, laid out with [`Vertex::layout`].
    pub vertex_buffer: wgpu::Buffer,
    /// Triangle-list indices into `vertex_buffer`.
    pub index_buffer: wgpu::Buffer,
    /// Number of indices — i.e. `3 * triangle count`, not the vertex count.
    pub index_count: u32,
    /// Minimum corner of the object-space bounding box.
    pub min: glam::Vec3,
    /// Maximum corner of the object-space bounding box.
    pub max: glam::Vec3,
}

impl Mesh {
    /// Upload a vertex/index pair to the GPU and compute its bounding box.
    ///
    /// Both buffers are created with `COPY_DST` in addition to their bind usage, so a caller that
    /// really does need to mutate a mesh (a deforming skin, say) can write into them instead of
    /// building a second `Mesh`.
    pub fn new(device: &wgpu::Device, name: &str, vertices: &[Vertex], indices: &[u32]) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{name}-vertices")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{name}-indices")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        let (min, max) = bounds(vertices);
        Self {
            name: name.to_string(),
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            min,
            max,
        }
    }
}

/// Axis-aligned bounds of a vertex list; an empty mesh collapses to the origin.
///
/// Seeding with `f32::MAX`/`f32::MIN` rather than with the first vertex means an empty list still
/// returns a valid (degenerate) box, and the renderer's culling turns a zero-radius sphere at the
/// origin into a cheap "probably out of view" answer instead of a panic.
fn bounds(vertices: &[Vertex]) -> (glam::Vec3, glam::Vec3) {
    if vertices.is_empty() {
        return (glam::Vec3::ZERO, glam::Vec3::ZERO);
    }
    let mut min = glam::Vec3::splat(f32::MAX);
    let mut max = glam::Vec3::splat(f32::MIN);
    for v in vertices {
        let p = glam::Vec3::from(v.position);
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}
