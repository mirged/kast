use anyhow::Result;
use glam::{Mat4, Vec3};
use std::f32::consts::PI;

use crate::geometry::{self, Geometry};
use crate::material::{Material, MaterialDesc, MaterialMaps};
use crate::mesh::Mesh;

pub mod loader;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LightKind {
    Directional,
    Point,
    Spot,
}

#[derive(Clone, Debug)]
pub struct Light {
    pub kind: LightKind,
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
    #[allow(dead_code)]
    pub cast_shadows: bool,
    pub enabled: bool,
}

impl Light {
    pub fn directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: direction.normalize(),
            color,
            intensity,
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cast_shadows: true,
            enabled: true,
        }
    }

    pub fn point(position: Vec3, color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            kind: LightKind::Point,
            position,
            direction: Vec3::NEG_Y,
            color,
            intensity,
            range,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cast_shadows: false,
            enabled: true,
        }
    }

    pub fn spot(
        position: Vec3,
        direction: Vec3,
        color: Vec3,
        intensity: f32,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            kind: LightKind::Spot,
            position,
            direction: direction.normalize(),
            color,
            intensity,
            range,
            inner_angle,
            outer_angle,
            cast_shadows: false,
            enabled: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sky {
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub sun_intensity: f32,
    pub turbidity: f32,
    pub ground_color: Vec3,
    pub exposure: f32,
    /// Sky colour straight up (linear HDR).
    pub zenith_color: Vec3,
    /// Sky colour at the horizon (linear HDR).
    pub horizon_color: Vec3,
    /// Exponent shaping the zenith→horizon gradient (higher = thinner horizon band).
    pub horizon_falloff: f32,
}

impl Default for Sky {
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(-0.35, 0.35, -0.6).normalize(),
            sun_color: Vec3::new(1.0, 0.86, 0.66),
            sun_intensity: 4.0,
            turbidity: 2.4,
            ground_color: Vec3::new(0.12, 0.11, 0.1),
            exposure: 1.0,
            zenith_color: Vec3::new(0.09, 0.19, 0.42),
            horizon_color: Vec3::new(0.62, 0.70, 0.82),
            horizon_falloff: 5.0,
        }
    }
}

pub struct SceneObject {
    pub mesh: usize,
    pub material: usize,
    pub transform: Mat4,
    pub visible: bool,
}

impl SceneObject {
    pub fn new(mesh: usize, material: usize, transform: Mat4) -> Self {
        Self {
            mesh,
            material,
            transform,
            visible: true,
        }
    }
}

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub objects: Vec<SceneObject>,
    pub lights: Vec<Light>,
    pub sky: Sky,
}

impl Scene {
    pub fn add_mesh(&mut self, device: &wgpu::Device, name: &str, geometry: Geometry) -> usize {
        let mesh = Mesh::new(device, name, &geometry.vertices, &geometry.indices);
        self.meshes.push(mesh);
        self.meshes.len() - 1
    }

    pub fn add_material(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: MaterialDesc,
        maps: MaterialMaps,
    ) -> Result<usize> {
        let material = Material::new(device, queue, desc, maps)?;
        self.materials.push(material);
        Ok(self.materials.len() - 1)
    }

    pub fn sun(&self) -> Option<&Light> {
        self.lights
            .iter()
            .find(|l| l.kind == LightKind::Directional)
    }
}

fn checker_texture(size: u32, cells: u32, a: [u8; 3], b: [u8; 3]) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(size, size);
    let cell = (size / cells.max(1)).max(1);
    for y in 0..size {
        for x in 0..size {
            let cx = (x / cell) % 2;
            let cy = (y / cell) % 2;
            let c = if (cx ^ cy) == 0 { a } else { b };
            img.put_pixel(x, y, image::Rgba([c[0], c[1], c[2], 255]));
        }
    }
    img
}

fn grid_texture(size: u32, base: [u8; 3], line: [u8; 3], line_width: u32) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(size, size);
    let cell = (size / 8).max(1);
    for y in 0..size {
        for x in 0..size {
            let on_line = (x % cell) < line_width || (y % cell) < line_width;
            let c = if on_line { line } else { base };
            let noise = ((x * 928371 + y * 1237) % 23) as i32 - 11;
            let adjust = |v: u8| (v as i32 + noise / 3).clamp(0, 255) as u8;
            img.put_pixel(
                x,
                y,
                image::Rgba([adjust(c[0]), adjust(c[1]), adjust(c[2]), 255]),
            );
        }
    }
    img
}

fn height_to_normal(height: &image::RgbaImage, strength: f32, tile: bool) -> image::RgbaImage {
    let w = height.width() as i32;
    let h = height.height() as i32;
    let mut out = image::RgbaImage::new(height.width(), height.height());
    let sample = |x: i32, y: i32| -> f32 {
        let mut xx = x;
        let mut yy = y;
        if tile {
            xx = xx.rem_euclid(w);
            yy = yy.rem_euclid(h);
        } else {
            xx = xx.clamp(0, w - 1);
            yy = yy.clamp(0, h - 1);
        }
        height.get_pixel(xx as u32, yy as u32)[0] as f32 / 255.0
    };
    for y in 0..h {
        for x in 0..w {
            let dx = sample(x + 1, y) - sample(x - 1, y);
            let dy = sample(x, y + 1) - sample(x, y - 1);
            let n = Vec3::new(-dx * strength, -dy * strength, 1.0).normalize();
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgba([
                    ((n.x * 0.5 + 0.5) * 255.0) as u8,
                    ((n.y * 0.5 + 0.5) * 255.0) as u8,
                    ((n.z * 0.5 + 0.5) * 255.0) as u8,
                    255,
                ]),
            );
        }
    }
    out
}

fn brick_height(size: u32) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(size, size);
    let rows = 8;
    let cols = 4;
    let bh = size / rows;
    let bw = size / cols;
    for y in 0..size {
        for x in 0..size {
            let row = y / bh;
            let offset = if row % 2 == 0 { 0 } else { bw / 2 };
            let bx = (x + offset) % bw;
            let by = y % bh;
            let mortar = 3;
            let is_mortar = bx < mortar || by < mortar;
            let v = if is_mortar {
                40
            } else {
                let n = ((x * 2654435761u32 ^ y * 40503) % 32) as u8;
                150 + n
            };
            img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
        }
    }
    img
}

pub fn default_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Scene> {
    let mut scene = Scene {
        meshes: Vec::new(),
        materials: Vec::new(),
        objects: Vec::new(),
        lights: Vec::new(),
        sky: Sky::default(),
    };

    let plane = scene.add_mesh(device, "plane", geometry::plane(60.0, 8));
    let sphere = scene.add_mesh(device, "sphere", geometry::uv_sphere(1.0, 64, 48));
    let torus = scene.add_mesh(device, "torus", geometry::torus(1.0, 0.38, 96, 48));
    let cube = scene.add_mesh(device, "cube", geometry::cube(1.0));
    let cylinder = scene.add_mesh(device, "cylinder", geometry::cylinder(0.6, 2.0, 48, true));

    let checker = checker_texture(512, 8, [210, 205, 195], [40, 42, 48]);
    let floor_maps = MaterialMaps {
        albedo: Some(grid_texture(1024, [58, 60, 64], [180, 182, 186], 3)),
        metallic_roughness: Some(checker_texture(256, 32, [0, 0, 0], [90, 90, 90])),
        ..Default::default()
    };
    let floor_mat = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "floor".into(),
            base_color: [0.6, 0.6, 0.62, 1.0],
            roughness: 0.7,
            metallic: 0.05,
            uv_scale: [12.0, 12.0],
            ..Default::default()
        },
        floor_maps,
    )?;

    let checker_mat = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "checker".into(),
            base_color: [1.0, 1.0, 1.0, 1.0],
            roughness: 0.15,
            metallic: 0.0,
            uv_scale: [2.0, 2.0],
            ..Default::default()
        },
        MaterialMaps {
            albedo: Some(checker),
            ..Default::default()
        },
    )?;

    let gold = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "gold".into(),
            base_color: [1.0, 0.766, 0.336, 1.0],
            metallic: 1.0,
            roughness: 0.13,
            ..Default::default()
        },
        MaterialMaps::default(),
    )?;

    let copper = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "copper".into(),
            base_color: [0.955, 0.637, 0.538, 1.0],
            metallic: 1.0,
            roughness: 0.32,
            ..Default::default()
        },
        MaterialMaps::default(),
    )?;

    let chrome = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "chrome".into(),
            base_color: [0.92, 0.93, 0.95, 1.0],
            metallic: 1.0,
            roughness: 0.05,
            ..Default::default()
        },
        MaterialMaps::default(),
    )?;

    let plastic = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "red-plastic".into(),
            base_color: [0.55, 0.045, 0.05, 1.0],
            metallic: 0.0,
            roughness: 0.32,
            ..Default::default()
        },
        MaterialMaps::default(),
    )?;

    let brick = brick_height(512);
    let brick_normal = height_to_normal(&brick, 3.0, true);
    let brick_albedo = {
        let mut alb = image::RgbaImage::new(512, 512);
        for (x, y, p) in alb.enumerate_pixels_mut() {
            let v = brick.get_pixel(x, y)[0];
            let tint = if v < 60 {
                [
                    v.saturating_add(20),
                    v.saturating_add(16),
                    v.saturating_add(10),
                ]
            } else {
                [v, (v as f32 * 0.62) as u8, (v as f32 * 0.42) as u8]
            };
            *p = image::Rgba([tint[0], tint[1], tint[2], 255]);
        }
        alb
    };
    let brick_mat = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "brick".into(),
            base_color: [0.8, 0.45, 0.3, 1.0],
            roughness: 0.85,
            metallic: 0.0,
            uv_scale: [1.0, 1.0],
            ..Default::default()
        },
        MaterialMaps {
            albedo: Some(brick_albedo),
            normal: Some(brick_normal),
            ..Default::default()
        },
    )?;

    let emissive_map = {
        let mut img = image::RgbaImage::new(256, 256);
        for y in 0..256 {
            for x in 0..256 {
                let d = (((x as i32 - 128).pow(2) + (y as i32 - 128).pow(2)) as f32).sqrt() / 128.0;
                let v = (1.0 - d).clamp(0.0, 1.0);
                let c = (v * v * 255.0) as u8;
                img.put_pixel(
                    x,
                    y,
                    image::Rgba([c, (c as f32 * 0.45) as u8, (c as f32 * 0.12) as u8, 255]),
                );
            }
        }
        img
    };
    let emissive_mat = scene.add_material(
        device,
        queue,
        MaterialDesc {
            name: "emissive".into(),
            base_color: [0.05, 0.02, 0.01, 1.0],
            emissive: [1.0, 0.45, 0.12],
            emissive_strength: 6.0,
            roughness: 0.4,
            metallic: 0.0,
            ..Default::default()
        },
        MaterialMaps {
            emissive: Some(emissive_map),
            ..Default::default()
        },
    )?;

    scene.objects.push(SceneObject::new(
        plane,
        floor_mat,
        Mat4::from_scale_rotation_translation(Vec3::splat(1.0), glam::Quat::IDENTITY, Vec3::ZERO),
    ));

    let metals = [gold, copper, chrome, plastic];
    for i in 0..4 {
        let angle = i as f32 / 4.0 * 2.0 * PI;
        let pos = Vec3::new(angle.cos() * 3.2, 1.0, angle.sin() * 3.2);
        scene.objects.push(SceneObject::new(
            sphere,
            metals[i],
            Mat4::from_translation(pos) * Mat4::from_scale(Vec3::splat(1.0)),
        ));
    }

    scene.objects.push(SceneObject::new(
        sphere,
        checker_mat,
        Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
    ));

    scene.objects.push(SceneObject::new(
        torus,
        brick_mat,
        Mat4::from_rotation_translation(
            glam::Quat::from_rotation_x(PI * 0.5),
            Vec3::new(0.0, 1.6, 0.0),
        ),
    ));

    for i in 0..3 {
        let x = -2.5 + i as f32 * 2.5;
        scene.objects.push(SceneObject::new(
            cube,
            if i == 1 { brick_mat } else { floor_mat },
            Mat4::from_scale_rotation_translation(
                Vec3::splat(1.2),
                glam::Quat::from_rotation_y(i as f32 * 0.7),
                Vec3::new(x, 0.6, -5.0),
            ),
        ));
    }

    scene.objects.push(SceneObject::new(
        sphere,
        emissive_mat,
        Mat4::from_translation(Vec3::new(0.0, 2.6, 0.0)) * Mat4::from_scale(Vec3::splat(0.55)),
    ));

    scene.objects.push(SceneObject::new(
        cylinder,
        chrome,
        Mat4::from_translation(Vec3::new(5.5, 1.0, 3.0)),
    ));

    let sun_travel = -scene.sky.sun_direction;
    scene.lights.push(Light::directional(
        sun_travel,
        scene.sky.sun_color,
        scene.sky.sun_intensity,
    ));
    scene.lights.push(Light::point(
        Vec3::new(4.0, 3.5, 4.0),
        Vec3::new(0.15, 0.45, 1.0),
        18.0,
        18.0,
    ));
    scene.lights.push(Light::point(
        Vec3::new(-4.5, 2.5, -2.0),
        Vec3::new(1.0, 0.25, 0.1),
        14.0,
        16.0,
    ));
    scene.lights.push(Light::spot(
        Vec3::new(0.0, 6.0, 6.0),
        Vec3::new(0.0, -1.0, -0.9),
        Vec3::new(1.0, 1.0, 1.0),
        120.0,
        30.0,
        (0.30_f32).cos(),
        (0.45_f32).cos(),
    ));

    Ok(scene)
}
