//! The scene: meshes, materials, objects, lights and sky — plain data the renderer reads.
//!
//! A [`Scene`] owns its GPU resources (uploaded meshes and materials) but is otherwise an ordinary
//! struct with public `Vec`s. That is deliberate: an application mutates `objects`, `lights` and
//! `sky` freely between frames, and the renderer picks the changes up with
//! [`crate::renderer::Renderer::update_objects`] without rebuilding a single pipeline.
//!
//! [`default_scene`] builds the demo scene the `kast` viewer shows — a shadowed floor, metal and
//! dielectric spheres, brick and emissive test materials, a sun and three extra lights — entirely
//! from procedural textures, so the crate ships with no assets.

use anyhow::Result;
use glam::{Mat4, Vec3};
use std::f32::consts::PI;

use crate::geometry::{self, Geometry};
use crate::material::{Material, MaterialDesc, MaterialMaps};
use crate::mesh::Mesh;

pub mod loader;

/// What a light's direction and position mean.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LightKind {
    /// Infinite sun: `direction` only, no attenuation. Exactly one is used for the shadow cascades.
    Directional,
    /// Omnidirectional point light with a soft `range` falloff.
    Point,
    /// Point light restricted to a cone by `inner_angle`/`outer_angle`.
    Spot,
}

/// One punctual light.
///
/// Construct with [`Light::directional`], [`Light::point`] or [`Light::spot`], which set the
/// fields a given kind actually uses; the struct keeps all of them so switching a light's kind or
/// animating it in place needs no reallocation of the scene's light list.
pub struct Light {
    /// Which of the three light types this is.
    pub kind: LightKind,
    /// World-space position. Unused (and zero) for directional lights.
    pub position: Vec3,
    /// Unit vector the light travels along — point it *away* from the lamp, toward what it lights.
    /// For the sun this is the reverse of [`Sky::sun_direction`].
    pub direction: Vec3,
    /// Linear RGB colour of the light.
    pub color: Vec3,
    /// Radiance multiplier applied to `color`. The unit is not physical — a sun of 4 and a point
    /// light of 18 are tuned against the sky's own intensity.
    pub intensity: f32,
    /// Clip distance in metres; lights beyond it contribute nothing. Ignored for directional
    /// lights.
    pub range: f32,
    /// Cosine of the spot's full-brightness cone angle (not the angle itself).
    pub inner_angle: f32,
    /// Cosine of the spot's outer cone angle; the falloff goes from `inner_angle` to here.
    pub outer_angle: f32,
    /// Requested shadow casting. Only the sun currently has a shadow map, so this is recorded but
    /// does not yet add punctual shadows.
    #[allow(dead_code)]
    pub cast_shadows: bool,
    /// Cheap on/off switch — a disabled light still occupies its slot but contributes nothing.
    pub enabled: bool,
}

impl Light {
    /// A sun: parallel light travelling along `direction`, with no distance falloff.
    ///
    /// `direction` is normalised here, and the light is created with `cast_shadows` set — it is
    /// the light the cascades are built from.
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

    /// An omnidirectional light at `position` reaching `range` metres, intensity falling off
    /// smoothly to zero at that distance.
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

    /// A cone light at `position` aimed along `direction`.
    ///
    /// `inner_angle` and `outer_angle` are the *cosines* of the cone half-angles, so 1.0 is a
    /// pencil-thin beam and lower values widen it — pass `(0.30_f32).cos()` for a 30° corner.
    /// Between the two cosines the light fades out, which is what gives a spotlight its soft edge.
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

/// The procedural sky: gradient colours, sun disc and the ground half.
///
/// The sky is rendered directly (it is the background) *and* used to build the image-based
/// lighting cubemaps, so changing these values relights the whole scene — ambient colour,
/// reflections and fog tint all follow from it.
pub struct Sky {
    /// Unit vector toward the sun. `default_scene` aims the directional light back along it.
    pub sun_direction: Vec3,
    /// Linear sun colour, also used for the disc and the IBL prefilter.
    pub sun_color: Vec3,
    /// Sun radiance multiplier; 4.0 is a bright but not blown-out afternoon.
    pub sun_intensity: f32,
    /// Haziness scalar kept for the analytic sky model; higher thickens the horizon band.
    pub turbidity: f32,
    /// Colour of the lower half of the environment, blended in at the horizon.
    pub ground_color: Vec3,
    /// Exposure applied while baking the sky cubemap (independent of the camera's EV).
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

/// One instance of a mesh with a material: the unit the renderer draws.
///
/// Objects are indices into [`Scene::meshes`] and [`Scene::materials`] rather than owners, so
/// instancing a model a hundred times costs a hundred small structs and nothing else.
pub struct SceneObject {
    /// Index into [`Scene::meshes`].
    pub mesh: usize,
    /// Index into [`Scene::materials`].
    pub material: usize,
    /// Model matrix: object space to world space. The renderer derives the normal matrix from
    /// its upper 3×3, so non-uniform scale is supported (normals are renormalised).
    pub transform: Mat4,
    /// Hidden objects are skipped by every pass but stay in the scene (and keep their buffers).
    pub visible: bool,
    /// Per-object opacity multiplier. Below 1 the object is drawn with ordered-dither
    /// (screen-door) transparency in the deferred pass.
    pub alpha: f32,
    /// Whether the object is rendered into the shadow cascades.
    pub cast_shadows: bool,
}

impl SceneObject {
    /// A fully visible, fully opaque, shadow-casting object at `transform`.
    pub fn new(mesh: usize, material: usize, transform: Mat4) -> Self {
        Self {
            mesh,
            material,
            transform,
            visible: true,
            alpha: 1.0,
            cast_shadows: true,
        }
    }
}

/// Everything that can be drawn, plus the lighting and sky it is drawn under.
///
/// Build one by hand (the fields are public and [`Scene::add_mesh`] / [`Scene::add_material`]
/// return the indices objects need), from [`default_scene`], or from a file with
/// [`loader::load`].
pub struct Scene {
    /// Uploaded meshes, indexed by [`SceneObject::mesh`].
    pub meshes: Vec<Mesh>,
    /// Uploaded materials, indexed by [`SceneObject::material`].
    pub materials: Vec<Material>,
    /// Drawable instances.
    pub objects: Vec<SceneObject>,
    /// Punctual lights; the first directional light is treated as the sun.
    pub lights: Vec<Light>,
    /// The sky dome, which also seeds the IBL cubemaps.
    pub sky: Sky,
}

impl Scene {
    /// Upload `geometry` as a new mesh and return its index.
    ///
    /// The geometry is consumed: its vertex/index buffers live on the GPU from here on.
    pub fn add_mesh(&mut self, device: &wgpu::Device, name: &str, geometry: Geometry) -> usize {
        let mesh = Mesh::new(device, name, &geometry.vertices, &geometry.indices);
        self.meshes.push(mesh);
        self.meshes.len() - 1
    }

    /// Create a material from `desc` and `maps` and return its index.
    ///
    /// Fails only if a texture upload fails. The bind group is created lazily by the renderer, so
    /// the material is usable the very next frame without any extra call.
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

    /// The first directional light, i.e. the one driving the shadow cascades.
    ///
    /// `None` when the scene has no sun, in which case shadows are simply not rendered and the
    /// sky's own sun position still lights the IBL.
    pub fn sun(&self) -> Option<&Light> {
        self.lights
            .iter()
            .find(|l| l.kind == LightKind::Directional)
    }
}

/// An RGBA checkerboard, used here for the tile pattern and the floor's roughness variation.
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

/// An 8×8 panel grid with faint per-pixel noise, so the floor's UVs are visible and the specular
/// response is not perfectly uniform.
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

/// Convert a grayscale height field into a tangent-space normal map by central differences.
///
/// `strength` scales the slope before normalisation. With `tile` the sampling wraps with
/// `rem_euclid`, so a seamless height map stays seamless — without it, the border is clamped,
/// which is what you want for a one-off photo-derived height map.
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

/// A running-bond brick height field: recessed mortar lines and per-brick value noise.
///
/// The noisy term uses `wrapping_mul` deliberately. In a release build the multiply wraps anyway,
/// but a debug build panics on overflow, so the explicit wrap is what keeps `cargo run` (and every
/// test) able to build the demo scene at all.
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
                let n = (x.wrapping_mul(2654435761u32) ^ y.wrapping_mul(40503)) % 32;
                150 + n as u8
            };
            img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
        }
    }
    img
}

/// Build the demo scene: a 60 m floor with a panel grid, four metal/dielectric spheres, a checker
/// sphere, a brick torus and cubes, an emissive sphere, a chrome cylinder, one sun and three
/// punctual lights.
///
/// Every texture is generated in code — checkerboards, a grid, brick height→normal, an emissive
/// radial gradient — so this is also a worked example of building materials without assets, and
/// it is what every screenshot in the README was rendered from.
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

    // The sun light travels along the reverse of the sky's sun direction, so the bright point in
    // the sky and the direction shadows are cast from agree.
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
