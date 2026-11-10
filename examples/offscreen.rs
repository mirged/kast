//! Render the built-in demo scene offscreen and write it to a PNG.
//!
//! ```sh
//! cargo run --example offscreen -- frame.png
//! ```
//!
//! This is the shortest path through the library API: build a scene, create the renderer, draw
//! frames, read the result back from the HDR present target.

use anyhow::{Context, Result};
use kast::camera::Camera;
use kast::context::GpuContext;
use kast::renderer::Renderer;
use kast::scene;

fn main() -> Result<()> {
    env_logger::init();
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "frame.png".into());
    let (width, height) = (1280, 720);

    let ctx = pollster::block_on(GpuContext::new_headless(width, height))?;
    let mut scene = scene::default_scene(&ctx.device, &ctx.queue)?;
    let mut renderer = Renderer::new(
        &ctx.device,
        &ctx.queue,
        width,
        height,
        ctx.target_format,
        ctx.srgb_target,
        true,
        &mut scene,
    )?;

    // Tune the look through `renderer.settings` — every field is live.
    renderer.settings.exposure_ev = 0.3;
    renderer.settings.fog_enabled = false;

    let camera = Camera::default();
    for _ in 0..4 {
        renderer.render(&ctx.device, &ctx.queue, &scene, &camera, None, 1.0 / 60.0);
    }
    ctx.device.poll(kast::wgpu::Maintain::Wait);

    let present = renderer
        .present
        .as_ref()
        .context("headless present target")?;
    let pixels = readback(&ctx, &present.texture, present.width, present.height);
    image::save_buffer(
        &path,
        &pixels,
        present.width,
        present.height,
        image::ColorType::Rgba8,
    )
    .with_context(|| format!("failed to write {path}"))?;
    log::info!("wrote {path}");
    Ok(())
}

/// Copy an RGBA8 texture back to the CPU, undoing the 256-byte row padding wgpu requires.
fn readback(ctx: &GpuContext, texture: &kast::wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
    let unpadded = width * 4;
    let padded = kast::gpu::align_to(unpadded, kast::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = ctx.device.create_buffer(&kast::wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * height) as u64,
        usage: kast::wgpu::BufferUsages::COPY_DST | kast::wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&kast::wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
    encoder.copy_texture_to_buffer(
        kast::wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: kast::wgpu::Origin3d::ZERO,
            aspect: kast::wgpu::TextureAspect::All,
        },
        kast::wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: kast::wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        kast::wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(kast::wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(kast::wgpu::Maintain::Wait);
    let _ = rx.recv();

    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    out
}
