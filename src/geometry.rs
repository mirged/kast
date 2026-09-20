use glam::{Vec2, Vec3, Vec4};
use std::f32::consts::PI;

use crate::mesh::Vertex;

pub struct Geometry {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Geometry {
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

    pub fn from_vertices(mut vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        compute_tangents(&mut vertices, &indices);
        Self { vertices, indices }
    }
}

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
