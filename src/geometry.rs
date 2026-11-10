//! Procedural primitives and the CPU-side geometry representation the renderer consumes.
//!
//! Every primitive is generated as an indexed triangle mesh with positions, normals, UVs and
//! tangents; [`Geometry`] is the hand-off type between generation (or a glTF importer) and
//! [`crate::mesh::Mesh`], which uploads it to the GPU.
//!
//! Tangents are derived from the UV layout rather than authored, so textures that rely on normal
//! mapping line up with the unwrap even for procedural shapes. Surfaces with degenerate UVs (poles
//! of a sphere, the seam of a cylinder) fall back to an arbitrary tangent orthogonal to the normal
//! and a `w` of ±1 that keeps the bitangent consistent across the seam.

use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

use crate::mesh::Vertex;

/// An indexed triangle mesh held in main memory.
///
/// Produced by the primitive constructors in this module, by
/// [`Geometry::from_parts`] for imported data, and uploaded once by
/// [`crate::mesh::Mesh::new`]. The renderer treats it as static: dynamic objects move by
/// per-object transforms, not by rewriting vertex buffers.
pub struct Geometry {
    /// Interleaved vertex records; `indices` index into this vector.
    pub vertices: Vec<Vertex>,
    /// Triangle list: every three entries form one triangle, wound counter-clockwise when seen
    /// from the outside (the front face).
    pub indices: Vec<u32>,
}

impl Geometry {
    /// Build geometry from separate attribute arrays, computing tangents.
    ///
    /// `positions`, `normals` and `uvs` must all be the same length and indexed together by
    /// `indices`; this is the layout glTF and most exporters use. Tangents are generated with
    /// `compute_tangents`, so callers never supply them.
    pub fn from_parts(
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        uvs: &[[f32; 2]],
        indices: Vec<u32>,
    ) -> Self {
        let mut vertices: Vec<Vertex> = (0..positions.len())
            .map(|i| Vertex {
                position: positions[i],
                normal: normals[i],
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv: uvs[i],
            })
            .collect();
        compute_tangents(&mut vertices, &indices);
        Self { vertices, indices }
    }

    /// Build geometry from ready-made vertices, recomputing their tangents.
    ///
    /// Any tangent already stored in `vertices` is discarded: normals, UVs and triangles are the
    /// authority, which keeps imported meshes consistent with the procedural ones.
    pub fn from_vertices(mut vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        compute_tangents(&mut vertices, &indices);
        Self { vertices, indices }
    }
}

/// Compute per-vertex tangents from positions, normals, UVs and triangles.
///
/// Uses the standard per-triangle tangent/bitangent accumulation: for each triangle the
/// derivatives of position and UV along the two edges give a tangent and a bitangent, which are
/// summed into its three vertices. The accumulated tangent is then Gram-Schmidt orthogonalised
/// against the vertex normal and its handedness is stored in `tangent.w`, so a shader can rebuild
/// the bitangent as `cross(normal, tangent.xyz) * tangent.w`.
///
/// Degenerate cases — zero-area UV triangles, vertices whose accumulated tangent cancels out, and
/// normals that are parallel to the fallback axis — are handled explicitly rather than producing
/// NaNs, because a single NaN vertex spreads across a whole draw call once interpolated.
fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut tan = vec![Vec3::ZERO; vertices.len()];
    let mut bitan = vec![Vec3::ZERO; vertices.len()];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let p0 = Vec3::from(vertices[i0].position);
        let p1 = Vec3::from(vertices[i1].position);
        let p2 = Vec3::from(vertices[i2].position);
        let uv0 = Vec2::from(vertices[i0].uv);
        let uv1 = Vec2::from(vertices[i1].uv);
        let uv2 = Vec2::from(vertices[i2].uv);
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let du1 = uv1 - uv0;
        let du2 = uv2 - uv0;
        let det = du1.x * du2.y - du1.y * du2.x;
        if det.abs() < 1e-8 {
            continue;
        }
        let r = 1.0 / det;
        let t = (e1 * du2.y - e2 * du1.y) * r;
        let b = (e2 * du1.x - e1 * du2.x) * r;
        tan[i0] += t;
        tan[i1] += t;
        tan[i2] += t;
        bitan[i0] += b;
        bitan[i1] += b;
        bitan[i2] += b;
    }

    for (i, v) in vertices.iter_mut().enumerate() {
        let n = Vec3::from(v.normal);
        let mut t = tan[i];
        if t.length_squared() < 1e-10 {
            t = Vec3::new(1.0, 0.0, 0.0);
        }
        t = (t - n * n.dot(t)).normalize_or_zero();
        if t.length_squared() < 1e-10 {
            t = if n.y.abs() < 0.99 {
                n.cross(Vec3::Y).normalize()
            } else {
                n.cross(Vec3::X).normalize()
            };
        }
        let b = bitan[i];
        let w = if n.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };
        v.tangent = Vec4::new(t.x, t.y, t.z, w).into();
    }
}

/// A flat grid in the XZ plane, centred on the origin and facing +Y.
///
/// `size` is the full edge length and `subdivisions` the number of quads per axis (clamped to at
/// least 1). UVs run 0..1 across the whole plane, so texture scale is controlled by the material's
/// `uv_scale` rather than by tessellation density.
pub fn plane(size: f32, subdivisions: u32) -> Geometry {
    let n = subdivisions.max(1);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let half = size * 0.5;
    for z in 0..=n {
        for x in 0..=n {
            let fx = x as f32 / n as f32;
            let fz = z as f32 / n as f32;
            positions.push([-half + fx * size, 0.0, -half + fz * size]);
            normals.push([0.0, 1.0, 0.0]);
            uvs.push([fx, fz]);
        }
    }
    let stride = n + 1;
    for z in 0..n {
        for x in 0..n {
            let i = z * stride + x;
            indices.extend_from_slice(&[i, i + stride, i + 1, i + 1, i + stride, i + stride + 1]);
        }
    }
    Geometry::from_parts(&positions, &normals, &uvs, indices)
}

/// An axis-aligned cube centred on the origin with edge length `size`.
///
/// Each face gets its own four vertices so the normals stay hard-edged, and UVs are oriented the
/// same way as the face's local right/up axes so image textures are not mirrored on opposite
/// faces.
pub fn cube(size: f32) -> Geometry {
    let h = size * 0.5;
    let faces: [([f32; 3], [f32; 3], [f32; 3], [f32; 3]); 6] = [
        (
            [0.0, 0.0, h],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ),
        (
            [0.0, 0.0, -h],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ),
        (
            [h, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ),
        (
            [-h, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ),
        (
            [0.0, h, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ),
        (
            [0.0, -h, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ),
    ];
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for (center, normal, right, up) in faces {
        let base = positions.len() as u32;
        let c = Vec3::from(center);
        let r = Vec3::from(right) * h;
        let u = Vec3::from(up) * h;
        for (su, sv, uv) in [
            (-1.0, -1.0, [0.0, 1.0]),
            (1.0, -1.0, [1.0, 1.0]),
            (1.0, 1.0, [1.0, 0.0]),
            (-1.0, 1.0, [0.0, 0.0]),
        ] {
            let p = c + r * su + u * sv;
            positions.push(p.into());
            normals.push(normal);
            uvs.push(uv);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Geometry::from_parts(&positions, &normals, &uvs, indices)
}

/// A latitude/longitude sphere of the given `radius`.
///
/// `segments` is the subdivision count around the equator and `rings` the count from pole to pole
/// (clamped to 3 and 2 respectively). Vertices sit exactly on the sphere and carry the sphere
/// normal, so the silhouette is smooth-shaded — but the poles are triangles with collapsed UVs,
/// which is exactly the degenerate case `compute_tangents` falls back on.
pub fn uv_sphere(radius: f32, segments: u32, rings: u32) -> Geometry {
    let segments = segments.max(3);
    let rings = rings.max(2);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for y in 0..=rings {
        let v = y as f32 / rings as f32;
        let theta = v * PI;
        for x in 0..=segments {
            let u = x as f32 / segments as f32;
            let phi = u * 2.0 * PI;
            let n = Vec3::new(
                theta.sin() * phi.sin(),
                theta.cos(),
                theta.sin() * phi.cos(),
            );
            positions.push((n * radius).into());
            normals.push(n.into());
            uvs.push([u, v]);
        }
    }
    let stride = segments + 1;
    for y in 0..rings {
        for x in 0..segments {
            let i = y * stride + x;
            indices.extend_from_slice(&[i, i + stride, i + 1, i + 1, i + stride, i + stride + 1]);
        }
    }
    Geometry::from_parts(&positions, &normals, &uvs, indices)
}

/// A torus lying in the XZ plane: `major` is the distance from the origin to the tube centre and
/// `minor` the tube radius.
///
/// The normal at every vertex points away from the tube centre line, so the inner surface (which
/// faces inward toward the hole) shades correctly instead of being lit from outside.
pub fn torus(major: f32, minor: f32, major_segments: u32, minor_segments: u32) -> Geometry {
    let major_segments = major_segments.max(3);
    let minor_segments = minor_segments.max(3);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for i in 0..=major_segments {
        let u = i as f32 / major_segments as f32;
        let phi = u * 2.0 * PI;
        let center = Vec3::new(phi.cos() * major, 0.0, phi.sin() * major);
        for j in 0..=minor_segments {
            let v = j as f32 / minor_segments as f32;
            let theta = v * 2.0 * PI;
            let normal = Vec3::new(
                phi.cos() * theta.cos(),
                theta.sin(),
                phi.sin() * theta.cos(),
            );
            let pos = center + normal * minor;
            positions.push(pos.into());
            normals.push(normal.into());
            uvs.push([u, v]);
        }
    }
    let stride = minor_segments + 1;
    for i in 0..major_segments {
        for j in 0..minor_segments {
            let a = i * stride + j;
            let b = (i + 1) * stride + j;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    Geometry::from_parts(&positions, &normals, &uvs, indices)
}

/// An upright cylinder along the Y axis, centred on the origin.
///
/// `radius` and `height` are full extents; `segments` is the subdivision count around the
/// circumference. With `capped` the two circular ends are added as triangle fans with flat ±Y
/// normals, wound outward (the top fan is reversed relative to the bottom so back-face culling
/// works on both). Without caps the shape is an open tube and is visibly hollow from above or
/// below — useful for tests, rarely what you want on screen.
pub fn cylinder(radius: f32, height: f32, segments: u32, capped: bool) -> Geometry {
    let segments = segments.max(3);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let half = height * 0.5;

    for i in 0..=segments {
        let u = i as f32 / segments as f32;
        let phi = u * 2.0 * PI;
        let n = Vec3::new(phi.cos(), 0.0, phi.sin());
        positions.push([n.x * radius, -half, n.z * radius]);
        normals.push(n.into());
        uvs.push([u, 1.0]);
        positions.push([n.x * radius, half, n.z * radius]);
        normals.push(n.into());
        uvs.push([u, 0.0]);
    }
    for i in 0..segments {
        let a = i * 2;
        indices.extend_from_slice(&[a, a + 2, a + 1, a + 1, a + 2, a + 3]);
    }

    if capped {
        for (sign, y) in [(1.0, half), (-1.0, -half)] {
            let base = positions.len() as u32;
            positions.push([0.0, y, 0.0]);
            normals.push([0.0, sign, 0.0]);
            uvs.push([0.5, 0.5]);
            for i in 0..=segments {
                let u = i as f32 / segments as f32;
                let phi = u * 2.0 * PI;
                positions.push([phi.cos() * radius, y, phi.sin() * radius]);
                normals.push([0.0, sign, 0.0]);
                uvs.push([0.5 + phi.cos() * 0.5, 0.5 + phi.sin() * 0.5]);
            }
            for i in 0..segments {
                if sign > 0.0 {
                    indices.extend_from_slice(&[base, base + 1 + i, base + 2 + i]);
                } else {
                    indices.extend_from_slice(&[base, base + 2 + i, base + 1 + i]);
                }
            }
        }
    }
    Geometry::from_parts(&positions, &normals, &uvs, indices)
}
