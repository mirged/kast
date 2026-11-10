//! GPU bring-up: instance, adapter, device, queue and (for windowed use) the swap-chain surface.
//!
//! [`GpuContext`] is the lowest layer of the crate. It exists so that application code does not
//! have to repeat the `wgpu` boilerplate — adapter selection with a software-rasterizer fallback,
//! device limits sized for the renderer, surface format and present-mode choice, resize
//! handling — and so that headless (offscreen) and windowed rendering go through the same setup.

use anyhow::{Context, Result, anyhow};
use std::sync::Arc;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::gpu;

/// A live `wgpu` device with the adapter, queue and optional window surface that produced it.
///
/// Create one with [`GpuContext::new_windowed`] or [`GpuContext::new_headless`], then hand
/// `&ctx.device`, `&ctx.queue` and the target format to [`crate::renderer::Renderer::new`]. In
/// windowed mode the surface is configured here, so the caller only has to acquire frames itself.
pub struct GpuContext {
    /// The `wgpu` instance the adapter and surface came from; kept alive because the surface
    /// borrows it.
    #[allow(dead_code)]
    pub instance: wgpu::Instance,
    /// The selected physical device. Exposed for capability queries (formats, limits).
    #[allow(dead_code)]
    pub adapter: wgpu::Adapter,
    /// The logical device every resource is created on.
    pub device: wgpu::Device,
    /// The submission queue used for buffer writes and command buffers.
    pub queue: wgpu::Queue,
    /// Window surface, `None` in headless mode.
    pub surface: Option<wgpu::Surface<'static>>,
    /// Surface configuration (format, present mode, size), `None` in headless mode.
    pub config: Option<wgpu::SurfaceConfiguration>,
    /// Current drawable size. In windowed mode this tracks [`GpuContext::resize`].
    pub size: PhysicalSize<u32>,
    /// Format to render into: the surface format when windowed, `Rgba8Unorm` when headless.
    pub target_format: wgpu::TextureFormat,
    /// Whether `target_format` applies the sRGB transfer function on write. The renderer uses
    /// this to decide if it still has to encode gamma in the tonemap pass.
    pub srgb_target: bool,
    /// Name, type and backend of the adapter, for logging and diagnostics.
    #[allow(dead_code)]
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Create a device for rendering into `window`.
    ///
    /// Picks an adapter that can present to that window (preferring a discrete GPU, falling back
    /// to a low-power one), configures the surface with `choose_format` and
    /// `choose_present_mode`, and logs the chosen GPU. The window is kept alive by the caller;
    /// the surface itself holds a `'static` reference because `wgpu` keeps its own handle.
    pub async fn new_windowed(window: Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .context("failed to create wgpu surface")?;

        let adapter = request_adapter(&instance, Some(&surface)).await?;
        let (device, queue) = create_device(&adapter).await?;

        let caps = surface.get_capabilities(&adapter);
        let format = choose_format(&caps.formats);
        let srgb_target = format.is_srgb();
        let adapter_info = adapter.get_info();

        let size = window.inner_size();
        // A minimised window reports 0x0, which is an invalid surface extent on most backends.
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));

        let mut config = surface
            .get_default_config(&adapter, size.width, size.height)
            .ok_or_else(|| anyhow!("surface is not compatible with the selected adapter"))?;
        config.format = format;
        config.present_mode = choose_present_mode(&caps.present_modes);
        config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        surface.configure(&device, &config);

        log::info!(
            "GPU: {} ({:?}) backend={:?} format={:?}",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend,
            format
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: Some(surface),
            config: Some(config),
            size,
            target_format: format,
            srgb_target,
            adapter_info,
        })
    }

    /// Create a device with no window, rendering into offscreen targets of the given size.
    ///
    /// The target format is `Rgba8Unorm` — 8 bits per channel, not sRGB — so the tonemap pass
    /// does its own gamma encoding and a readback can be written straight to a PNG. This is the
    /// path used for screenshots, tests and CI; it still needs a working driver, since `wgpu` has
    /// no pure-CPU backend of its own.
    pub async fn new_headless(width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = request_adapter(&instance, None).await?;
        let (device, queue) = create_device(&adapter).await?;

        let size = PhysicalSize::new(width.max(1), height.max(1));
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let adapter_info = adapter.get_info();

        log::info!(
            "Headless GPU: {} ({:?}) backend={:?}",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface: None,
            config: None,
            size,
            target_format: format,
            srgb_target: false,
            adapter_info,
        })
    }

    /// Resize the drawable and reconfigure the surface.
    ///
    /// Zero in either axis is clamped to 1 so a minimised window does not produce an invalid
    /// surface extent. The renderer's own render targets are *not* resized here — call
    /// [`crate::renderer::Renderer::resize`] as well, with the same size.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let width = new_size.width.max(1);
        let height = new_size.height.max(1);
        self.size = PhysicalSize::new(width, height);
        if let (Some(surface), Some(config)) = (&self.surface, &mut self.config) {
            config.width = width;
            config.height = height;
            surface.configure(&self.device, config);
        }
    }
}

/// Find an adapter, preferring performance and falling back to a software rasterizer.
///
/// The second pass sets `force_fallback_adapter`, which is what makes headless rendering work on
/// machines with no GPU driver (Mesa `lavapipe`/`llvmpipe` report as low-power/CPU adapters).
/// Without it, a CI container fails at start-up even though a perfectly usable software device is
/// installed.
async fn request_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<wgpu::Adapter> {
    for force_fallback in [false, true] {
        if let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: if force_fallback {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                force_fallback_adapter: force_fallback,
                compatible_surface: surface,
            })
            .await
        {
            return Ok(adapter);
        }
    }
    Err(anyhow!(
        "no suitable GPU adapter found. Install a Vulkan/Metal/DX12 driver or use a software rasterizer."
    ))
}

/// Open a device on `adapter` with the limits the renderer needs.
///
/// No optional features are requested: everything the pipelines use (depth-stencil, cube/array
/// textures, storage buffers for the light list, 16x anisotropy on capable adapters) is core in
/// WebGPU, which keeps the crate running on the widest set of drivers.
async fn create_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let limits = gpu::gpu_limits_for(adapter);
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("kast-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .context("failed to create wgpu device")?;
    Ok((device, queue))
}

/// Pick the surface format: a non-sRGB one if the surface offers it, else whatever is first.
///
/// Non-sRGB is preferred because the tonemap pass already produces display-encoded values; using
/// an `*Srgb` surface would apply the transfer function a second time and wash the image out. If
/// only sRGB formats exist, [`GpuContext::srgb_target`] is set and the renderer skips its own
/// encoding instead.
fn choose_format(formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
    if let Some(f) = formats.iter().find(|f| !f.is_srgb()) {
        return *f;
    }
    formats
        .first()
        .copied()
        .unwrap_or(wgpu::TextureFormat::Bgra8Unorm)
}

/// Pick the present mode: mailbox (low latency, no tearing) if available, else FIFO vsync.
///
/// FIFO is the only mode WebGPU guarantees, so it is the deliberate fallback rather than
/// immediate, which tears.
fn choose_present_mode(modes: &[wgpu::PresentMode]) -> wgpu::PresentMode {
    if modes.contains(&wgpu::PresentMode::Mailbox) {
        wgpu::PresentMode::Mailbox
    } else if modes.contains(&wgpu::PresentMode::Fifo) {
        wgpu::PresentMode::Fifo
    } else {
        modes.first().copied().unwrap_or(wgpu::PresentMode::Fifo)
    }
}
