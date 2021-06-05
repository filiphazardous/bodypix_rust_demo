use nannou::image;
use nannou::image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};
use crate::bodypix::Segments;

pub fn mask_to_image(mask: &Segments) -> DynamicImage {
    let output_img = ImageBuffer::from_fn(mask.orig_width as u32, mask.orig_height as u32, |x, y| {
        let shade_val: u8 = (mask.linear_mean(x as usize, y as usize) * 255.) as u8;
        image::Rgb([shade_val, shade_val, shade_val])
    });

    DynamicImage::ImageRgb8(output_img)
}

pub fn create_silhouette(mask: &Segments, orig: &DynamicImage) -> DynamicImage {
    let width = orig.width();
    let height = orig.height();
    let black_pixel = image::Rgb([0, 0, 0]);
    let silhouette_img = ImageBuffer::from_fn(width, height, |x, y| {
        if mask.linear_mean(x as usize, y as usize) > 0.7 {
            orig.get_pixel(x, y).to_rgb()
        } else {
            black_pixel
        }
    });
    DynamicImage::ImageRgb8(silhouette_img)
}

pub fn create_cutout(mask: &Segments, orig: &DynamicImage) -> DynamicImage {
    let width = orig.width();
    let height = orig.height();
    let black_pixel = image::Rgb([0, 0, 0]);
    let silhouette_img = ImageBuffer::from_fn(width, height, |x, y| {
        if mask.linear_mean(x as usize, y as usize) <= 0.7 {
            orig.get_pixel(x, y).to_rgb()
        } else {
            black_pixel
        }
    });
    DynamicImage::ImageRgb8(silhouette_img)
}
