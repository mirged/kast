//! kast — a photorealistic real-time renderer built on [`wgpu`].
//!
//! The renderer is a deferred PBR pipeline: a geometry pass writes albedo, normal, emissive and
//! depth, then a lighting pass shades it from an image-based sky, up to 64 punctual lights and
//! four cascaded PCSS sun-shadow maps. Screen-space ambient occlusion, screen-space reflections,
//! bloom, depth of field, ACES tonemapping and FXAA finish the frame, with optional exponential
//! height fog and ordered-dither transparency.
//!
//! Rendering is split into three pieces that the caller owns:
//!
//! - a [`context::GpuContext`] holding the adapter, device, queue and (windowed) surface,
//! - a [`scene::Scene`] of meshes, materials, objects, lights and sky, built with
//!   [`scene::Scene::add_mesh`] / [`scene::Scene::add_material`] or loaded from glTF with
//!   [`scene::loader::load`],
//! - a [`renderer::Renderer`], which is created once and driven per frame with
//!   [`renderer::Renderer::render`]. Its [`renderer::Settings`] fields toggle and tune each effect
//!   at runtime.
//!
//! ```no_run
//! use kast::camera::Camera;
//! use kast::context::GpuContext;
//! use kast::renderer::Renderer;
//! use kast::scene;
//!
//! # fn main() -> anyhow::Result<()> {
//! let (width, height) = (1280, 720);
//! let ctx = pollster::block_on(GpuContext::new_headless(width, height))?;
//! let mut scene = scene::default_scene(&ctx.device, &ctx.queue)?;
//! let mut renderer = Renderer::new(
//!     &ctx.device, &ctx.queue, width, height,
//!     ctx.target_format, ctx.srgb_target, /* headless */ true, &mut scene,
//! )?;
//!
//! let camera = Camera::default();
//! renderer.render(&ctx.device, &ctx.queue, &scene, &camera, None, 1.0 / 60.0);
//! # Ok(())
//! # }
//! ```
//!
//! The `kast` binary in this package is a thin front end over the same API: it opens a window on
//! the built-in demo scene (or a glTF model) and can also render offscreen to a PNG.
pub mod camera;
pub mod context;
pub mod geometry;
pub mod gpu;
pub mod material;
pub mod mesh;
pub mod renderer;
pub mod scene;
pub mod types;

// Re-exported so downstream crates can build a `Scene`/`Renderer` against the exact same
// `wgpu` and `glam` versions instead of pinning their own.
pub use glam;
pub use wgpu;
