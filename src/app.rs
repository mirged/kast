//! The interactive viewer: window, event loop, and the per-frame hand-off to the renderer.
//!
//! This module is the binary's, not the library's — it exists so `main.rs` stays a thin argument
//! parser. [`App`] implements `winit`'s `ApplicationHandler`, owns the window and the renderer
//! state, and translates input into [`Camera`] calls and [`Settings`] toggles. The interesting
//! part for a library user is [`render_frame`]: acquire the swap-chain texture, render into it,
//! present, and cope with a lost surface.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use kast::camera::Camera;
use kast::context::GpuContext;
use kast::renderer::Renderer;
use kast::scene::{self, Scene};

/// The application: lazily-initialised GPU state plus the values that outlive it.
///
/// Nothing GPU-bound exists until the event loop reports `resumed`, which is why [`State`] is an
/// `Option` — on some platforms a window can be re-created after the surface is lost, and the
/// renderer has to be rebuilt with it.
pub struct App {
    /// Model path from `--model`, if the viewer was asked to load one; `None` means the
    /// procedural demo scene.
    pub model: Option<PathBuf>,
    /// The window, kept alive for as long as the surface borrows it.
    window: Option<Arc<Window>>,
    /// Device, renderer, scene and camera, created in `resumed`.
    state: Option<State>,
    /// Last cursor position, used to decide whether to grab the pointer.
    cursor: Option<PhysicalPosition<f64>>,
    /// Timestamp of the previous frame, for the delta passed to the camera and the renderer.
    last_frame: Instant,
    /// Frames presented since start-up, for the periodic fps log.
    frame_count: u64,
    /// Seconds accumulated since the last fps log.
    accum: f32,
    /// Frames counted in `accum`.
    samples: u32,
}

/// Everything that only exists once a window and a GPU are available.
struct State {
    /// Device, queue and the configured surface.
    ctx: GpuContext,
    /// All pipelines and render targets, sized to the surface.
    renderer: Renderer,
    /// The scene being viewed (demo or glTF) and the camera flying through it.
    scene: Scene,
    camera: Camera,
}

impl App {
    pub fn new(model: Option<PathBuf>) -> Self {
        Self {
            model,
            window: None,
            state: None,
            cursor: None,
            last_frame: Instant::now(),
            frame_count: 0,
            accum: 0.0,
            samples: 0,
        }
    }

    fn init_state(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("kast - wgpu photorealistic renderer")
                    .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0)),
            )?,
        );
        self.window = Some(window.clone());

        let ctx = pollster::block_on(GpuContext::new_windowed(window.clone()))?;
        let mut scene = match &self.model {
            Some(path) => scene::loader::load(path, &ctx.device, &ctx.queue)?,
            None => scene::default_scene(&ctx.device, &ctx.queue)?,
        };
        let mut camera = Camera::default();
        frame_scene(&mut camera, &scene);

        let renderer = Renderer::new(
            &ctx.device,
            &ctx.queue,
            ctx.size.width,
            ctx.size.height,
            ctx.target_format,
            ctx.srgb_target,
            false,
            &mut scene,
        )?;

        // Push the current camera state into the renderer once; from here on a redraw keeps the
        // two in sync (`exposure_ev` per frame, in `window_event`).
        self.state = Some(State {
            ctx,
            renderer,
            scene,
            camera,
        });
        self.last_frame = Instant::now();
        Ok(())
    }
}

/// Aim the camera at the whole scene and back off far enough to see it.
///
/// The bounds are the union of every object's transformed mesh bounding box, which is exactly what
/// the renderer's culler uses — so the initial framing is never inside a model, and a 2 m prop and
/// a 2 km island both come into view without hand-tuned numbers.
fn frame_scene(camera: &mut Camera, scene: &Scene) {
    let mut min = glam::Vec3::splat(f32::MAX);
    let mut max = glam::Vec3::splat(f32::MIN);
    for obj in &scene.objects {
        let mesh = &scene.meshes[obj.mesh];
        let a = obj.transform.transform_point3(mesh.min);
        let b = obj.transform.transform_point3(mesh.max);
        min = min.min(a).min(b);
        max = max.max(a).max(b);
    }
    if min.x <= max.x {
        let center = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        camera.target = center;
        camera.distance = (radius * 2.2).clamp(2.0, 200.0);
        camera.position = center + glam::Vec3::new(0.6, 0.5, 1.0).normalize() * camera.distance;
    }
}

impl ApplicationHandler for App {
    /// Create the window and all GPU state the first time the platform hands us an event loop.
    ///
    /// A failure here is fatal for the viewer: it is logged with its full context chain and the
    /// loop exits, rather than leaving a blank window up.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            if let Err(err) = self.init_state(event_loop) {
                log::error!("failed to initialize renderer: {err:#}");
                event_loop.exit();
            }
        }
    }

    /// Route a window event: quit keys, the F-key effect toggles, camera input, resize, redraw.
    ///
    /// Function keys are handled here rather than in the camera because they mutate
    /// [`Settings`], which the camera does not know about; everything else is delegated. Resize
    /// updates the surface *and* the renderer's targets, which are two independent sizes to keep
    /// in step.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if self.window.as_ref().map(|w| w.id()) != Some(window_id) {
            return;
        }

        match &event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return;
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(KeyCode::Escape) = key_event.physical_key {
                    if key_event.state == ElementState::Pressed {
                        event_loop.exit();
                        return;
                    }
                }
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    if key_event.state == ElementState::Pressed {
                        match code {
                            KeyCode::F1 => {
                                state.renderer.settings.ssr_enabled =
                                    !state.renderer.settings.ssr_enabled;
                                log_state(state);
                            }
                            KeyCode::F2 => {
                                state.renderer.settings.ssao_enabled =
                                    !state.renderer.settings.ssao_enabled;
                                log_state(state);
                            }
                            KeyCode::F3 => {
                                state.renderer.settings.shadows_enabled =
                                    !state.renderer.settings.shadows_enabled;
                                log_state(state);
                            }
                            KeyCode::F4 => {
                                state.renderer.settings.ibl_enabled =
                                    !state.renderer.settings.ibl_enabled;
                                log_state(state);
                            }
                            KeyCode::F5 => {
                                state.renderer.settings.dof_enabled =
                                    !state.renderer.settings.dof_enabled;
                                log_state(state);
                            }
                            KeyCode::F6 => {
                                state.renderer.settings.fxaa_enabled =
                                    !state.renderer.settings.fxaa_enabled;
                                log_state(state);
                            }
                            KeyCode::F7 => {
                                state.renderer.settings.tonemap_mode =
                                    (state.renderer.settings.tonemap_mode + 1) % 3;
                                log::info!(
                                    "tonemap mode: {}",
                                    state.renderer.settings.tonemap_mode
                                );
                            }
                            KeyCode::F8 => {
                                state.renderer.settings.debug_ssao =
                                    !state.renderer.settings.debug_ssao;
                                log_state(state);
                            }
                            _ => {}
                        }
                    }
                }
                state.camera.process_event(&event);
            }
            WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                state.camera.process_event(&event);
            }
            WindowEvent::Resized(size) => {
                state.ctx.resize(*size);
                state
                    .renderer
                    .resize(&state.ctx.device, &state.ctx.queue, size.width, size.height);
            }
            WindowEvent::Focused(false) => {
                state.camera.process_event(&event);
                self.cursor = None;
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;
                state.camera.update(dt);
                state.renderer.settings.exposure_ev = state.camera.exposure_ev;

                if let Err(err) = render_frame(state) {
                    log::error!("render error: {err:#}");
                    event_loop.exit();
                    return;
                }

                self.frame_count += 1;
                self.accum += dt;
                self.samples += 1;
                if self.accum > 2.0 {
                    log::info!(
                        "fps: {:.1}  exposure_ev: {:.1}",
                        self.samples as f32 / self.accum,
                        state.camera.exposure_ev
                    );
                    self.accum = 0.0;
                    self.samples = 0;
                }
            }
            _ => {}
        }
    }

    /// Feed raw mouse motion to the camera.
    ///
    /// Raw deltas come through `DeviceEvent`, not `WindowEvent`, because the cursor is captured
    /// and its absolute position would stop at the window edge — raw motion is unbounded and gives
    /// a first-person camera its full range of travel.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let (Some(state), DeviceEvent::MouseMotion { delta }) = (self.state.as_mut(), event) {
            state
                .camera
                .process_mouse_motion(delta.0 as f32, delta.1 as f32);
        }
    }

    /// Request a redraw every time the loop goes idle, i.e. render at the present mode's rate.
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let (Some(window), Some(state)) = (&self.window, &self.state) {
            let _ = state;
            window.request_redraw();
        }
    }
}

/// Log the effect toggles, so an F-key press leaves a record of the resulting state.
fn log_state(state: &State) {
    let s = &state.renderer.settings;
    log::info!(
        "ssr={} ssao={} shadows={} ibl={} dof={} fxaa={} debug_ssao={}",
        s.ssr_enabled,
        s.ssao_enabled,
        s.shadows_enabled,
        s.ibl_enabled,
        s.dof_enabled,
        s.fxaa_enabled,
        s.debug_ssao
    );
}

/// Render one frame to the swap chain and present it.
///
/// The surface-error handling is the part worth copying into a host application: `Lost` and
/// `Outdated` mean the swap chain needs reconfiguring (resize, mode change, driver hiccup) and are
/// recoverable by skipping the frame; `Timeout` is likewise benign; only `OutOfMemory` is fatal.
fn render_frame(state: &mut State) -> Result<()> {
    let ctx = &mut state.ctx;
    let surface = ctx
        .surface
        .as_ref()
        .expect("windowed context has a surface");

    let frame = match surface.get_current_texture() {
        Ok(frame) => frame,
        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
            if let Some(config) = &ctx.config {
                surface.configure(&ctx.device, config);
            }
            return Ok(());
        }
        Err(wgpu::SurfaceError::OutOfMemory) => anyhow::bail!("surface out of memory"),
        Err(wgpu::SurfaceError::Timeout) => return Ok(()),
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    state.renderer.render(
        &ctx.device,
        &ctx.queue,
        &state.scene,
        &state.camera,
        Some(&view),
        1.0 / 60.0,
    );

    frame.present();
    Ok(())
}
