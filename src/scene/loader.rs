use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use std::collections::HashMap;
use std::path::Path;

use super::{Light, Scene, SceneObject};
use crate::geometry::Geometry;
use crate::material::{MaterialDesc, MaterialMaps};
use crate::mesh::Vertex;

pub fn load(path: &Path, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Scene> {
    let mut scene = Scene {
        meshes: Vec::new(),
        materials: Vec::new(),
        objects: Vec::new(),
        lights: Vec::new(),
        sky: super::Sky::default(),
    };
    load_into(path, device, queue, &mut scene)?;
    Ok(scene)
}

pub fn load_into(
    path: &Path,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &mut Scene,
) -> Result<()> {
    let (document, buffers, images) =
        gltf::import(path).with_context(|| format!("failed to import glTF: {}", path.display()))?;

    let mut material_map: HashMap<usize, usize> = HashMap::new();
    for (index, material) in document.materials().enumerate() {
        let (desc, maps) = import_material(&material, &images);
        let id = scene.add_material(device, queue, desc, maps)?;
        material_map.insert(index, id);
    }

    let default_material = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "gltf-default".into(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            ..Default::default()
        },
        MaterialMaps::default(),
    )?;

    let mut mesh_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (index, mesh) in document.meshes().enumerate() {
        let mut ids = Vec::new();
        for primitive in mesh.primitives() {
            let geometry = read_primitive(&primitive, &buffers)?;
            let name = mesh.name().unwrap_or("gltf-mesh");
            ids.push(scene.add_mesh(device, name, geometry));
        }
        mesh_map.insert(index, ids);
    }

    let mut roots: Vec<gltf::Node> = Vec::new();
    if let Some(default) = document.default_scene() {
        roots.extend(default.nodes());
    } else if let Some(first) = document.scenes().next() {
        roots.extend(first.nodes());
    }
    for node in roots {
        node_walk(
            node,
            Mat4::IDENTITY,
            &mesh_map,
            &material_map,
            default_material,
            scene,
        );
    }

    if let Some(_lights) = document.lights() {
        // Node-referenced lights are handled during traversal.
    }

    Ok(())
}

fn node_walk(
    node: gltf::Node,
    parent: Mat4,
    mesh_map: &HashMap<usize, Vec<usize>>,
    material_map: &HashMap<usize, usize>,
    default_material: usize,
    out: &mut Scene,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        if let Some(primitives) = mesh_map.get(&mesh.index()) {
            let all: Vec<gltf::Primitive> = mesh.primitives().collect();
            for (prim_index, &mesh_id) in primitives.iter().enumerate() {
                let mat = all
                    .get(prim_index)
                    .and_then(|p| p.material().index())
                    .and_then(|i| material_map.get(&i).copied())
                    .unwrap_or(default_material);
                out.objects.push(SceneObject::new(mesh_id, mat, world));
            }
        }
    }

    if let Some(light) = node.light() {
        let color = Vec3::from(light.color());
        let intensity = light.intensity();
        let range = light.range().unwrap_or(20.0);
        let position = world.transform_point3(Vec3::ZERO);
        let forward = (-world.transform_vector3(Vec3::Z)).normalize_or_zero();
        match light.kind() {
            gltf::khr_lights_punctual::Kind::Directional => {
                out.lights
                    .push(Light::directional(forward, color, intensity));
            }
            gltf::khr_lights_punctual::Kind::Point => {
                out.lights
                    .push(Light::point(position, color, intensity, range));
            }
            gltf::khr_lights_punctual::Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => {
                out.lights.push(Light::spot(
                    position,
                    forward,
                    color,
                    intensity,
                    range,
                    inner_cone_angle.cos(),
                    outer_cone_angle.cos(),
                ));
            }
        }
    }

    for child in node.children() {
        node_walk(child, world, mesh_map, material_map, default_material, out);
    }
}

type Buffers = Vec<gltf::buffer::Data>;

fn read_primitive(primitive: &gltf::Primitive, buffers: &Buffers) -> Result<Geometry> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .context("primitive has no POSITION attribute")?
        .collect();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|n| n.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|t| t.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|t| t.collect());

    let indices: Vec<u32> = match reader.read_indices() {
        Some(i) => i.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };

    let vertices: Vec<Vertex> = (0..positions.len())
        .map(|i| Vertex {
            position: positions[i],
            normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            tangent: tangents
                .as_ref()
                .and_then(|t| t.get(i))
                .copied()
                .unwrap_or([1.0, 0.0, 0.0, 1.0]),
            uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
        })
        .collect();

    if tangents.is_some() {
        Ok(Geometry { vertices, indices })
    } else {
        Ok(Geometry::from_vertices(vertices, indices))
    }
}

fn import_material(
    material: &gltf::Material,
    images: &[gltf::image::Data],
) -> (MaterialDesc, MaterialMaps) {
    let pbr = material.pbr_metallic_roughness();
    let base = pbr.base_color_factor();
    let emissive = material.emissive_factor();
    let desc = MaterialDesc {
        name: material.name().unwrap_or("gltf-material").to_string(),
        base_color: base,
        emissive: [emissive[0], emissive[1], emissive[2]],
        emissive_strength: material.emissive_strength().unwrap_or(1.0),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        ao: 1.0,
        alpha_cutoff: if material.alpha_mode() == gltf::material::AlphaMode::Mask {
            0.5
        } else {
            0.0
        },
        uv_scale: [1.0, 1.0],
        uv_offset: [0.0, 0.0],
        double_sided: material.double_sided(),
        unlit: false,
    };

    let mut maps = MaterialMaps::default();
    if let Some(info) = pbr.base_color_texture() {
        maps.albedo = decode_image(info.texture().source().index(), images);
    }
    if let Some(info) = material.normal_texture() {
        maps.normal = decode_image(info.texture().source().index(), images);
    }
    if let Some(info) = pbr.metallic_roughness_texture() {
        maps.metallic_roughness = decode_image(info.texture().source().index(), images);
    }
    if let Some(info) = material.emissive_texture() {
        maps.emissive = decode_image(info.texture().source().index(), images);
    }
    if let Some(info) = material.occlusion_texture() {
        maps.occlusion = decode_image(info.texture().source().index(), images);
    }

    (desc, maps)
}

fn decode_image(index: usize, images: &[gltf::image::Data]) -> Option<image::RgbaImage> {
    let data = images.get(index)?;
    let w = data.width;
    let h = data.height;
    let pixels = &data.pixels;
    let mut out = image::RgbaImage::new(w, h);
    match data.format {
        gltf::image::Format::R8 => {
            for (i, p) in out.pixels_mut().enumerate() {
                let v = pixels[i];
                *p = image::Rgba([v, v, v, 255]);
            }
        }
        gltf::image::Format::R8G8 => {
            for (i, p) in out.pixels_mut().enumerate() {
                let v = pixels[i * 2];
                *p = image::Rgba([v, v, v, pixels[i * 2 + 1]]);
            }
        }
        gltf::image::Format::R8G8B8 => {
            for (i, p) in out.pixels_mut().enumerate() {
                *p = image::Rgba([pixels[i * 3], pixels[i * 3 + 1], pixels[i * 3 + 2], 255]);
            }
        }
        gltf::image::Format::R8G8B8A8 => {
            for (i, p) in out.pixels_mut().enumerate() {
                *p = image::Rgba([
                    pixels[i * 4],
                    pixels[i * 4 + 1],
                    pixels[i * 4 + 2],
                    pixels[i * 4 + 3],
                ]);
            }
        }
        gltf::image::Format::R16 => {
            for (i, p) in out.pixels_mut().enumerate() {
                let v = (u16::from_le_bytes([pixels[i * 2], pixels[i * 2 + 1]]) >> 8) as u8;
                *p = image::Rgba([v, v, v, 255]);
            }
        }
        gltf::image::Format::R16G16 => {
            for (i, p) in out.pixels_mut().enumerate() {
                let v = (u16::from_le_bytes([pixels[i * 4], pixels[i * 4 + 1]]) >> 8) as u8;
                *p = image::Rgba([v, v, v, 255]);
            }
        }
        gltf::image::Format::R16G16B16 => {
            for (i, p) in out.pixels_mut().enumerate() {
                let r = (u16::from_le_bytes([pixels[i * 6], pixels[i * 6 + 1]]) >> 8) as u8;
                let g = (u16::from_le_bytes([pixels[i * 6 + 2], pixels[i * 6 + 3]]) >> 8) as u8;
                let b = (u16::from_le_bytes([pixels[i * 6 + 4], pixels[i * 6 + 5]]) >> 8) as u8;
                *p = image::Rgba([r, g, b, 255]);
            }
        }
        gltf::image::Format::R16G16B16A16 => {
            for (i, p) in out.pixels_mut().enumerate() {
                let r = (u16::from_le_bytes([pixels[i * 8], pixels[i * 8 + 1]]) >> 8) as u8;
                let g = (u16::from_le_bytes([pixels[i * 8 + 2], pixels[i * 8 + 3]]) >> 8) as u8;
                let b = (u16::from_le_bytes([pixels[i * 8 + 4], pixels[i * 8 + 5]]) >> 8) as u8;
                let a = (u16::from_le_bytes([pixels[i * 8 + 6], pixels[i * 8 + 7]]) >> 8) as u8;
                *p = image::Rgba([r, g, b, a]);
            }
        }
        gltf::image::Format::R32G32B32FLOAT => {
            for (i, p) in out.pixels_mut().enumerate() {
                let v = |o: usize| {
                    f32::from_le_bytes([
                        pixels[i * 12 + o],
                        pixels[i * 12 + o + 1],
                        pixels[i * 12 + o + 2],
                        pixels[i * 12 + o + 3],
                    ])
                };
                *p = image::Rgba([
                    (v(0) * 255.0) as u8,
                    (v(4) * 255.0) as u8,
                    (v(8) * 255.0) as u8,
                    255,
                ]);
            }
        }
        gltf::image::Format::R32G32B32A32FLOAT => {
            for (i, p) in out.pixels_mut().enumerate() {
                let v = |o: usize| {
                    f32::from_le_bytes([
                        pixels[i * 16 + o],
                        pixels[i * 16 + o + 1],
                        pixels[i * 16 + o + 2],
                        pixels[i * 16 + o + 3],
                    ])
                };
                *p = image::Rgba([
                    (v(0) * 255.0) as u8,
                    (v(4) * 255.0) as u8,
                    (v(8) * 255.0) as u8,
                    (v(12) * 255.0) as u8,
                ]);
            }
        }
    }
    Some(out)
}
