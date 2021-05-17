use std::fs::File;
use std::io::Read;
use std::ops::Index;
use std::path::PathBuf;

use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionRunArgs, Tensor};

use nannou::image;
use nannou::image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};

pub fn load_bodypix_model(assets_path: PathBuf) -> (Graph, u32) {
    let model_file = assets_path
        .join("models")
        .join("bodypix_resnet50_float_model-stride16.pb");

    let stride: u32 = 16;

    let mut model_data: Vec<u8> = Vec::new();
    let mut file = File::open(model_file).unwrap();
    file.read_to_end(&mut model_data).unwrap();

    let mut graph = Graph::new();

    let import_results = graph
        .import_graph_def_with_results(&model_data, &ImportGraphDefOptions::new())
        .unwrap();

    println!("Results: \n");
    let outputs = import_results.return_outputs();
    println!("Return outputs: {}\n", outputs.len());
    for o in &outputs {
        println!(
            "  {}:{} Type: {}",
            o.operation.name().unwrap(),
            o.index,
            o.operation.op_type().unwrap()
        );
    }
    let operations = import_results.return_operations();
    println!("Return operations: {}\n", operations.len());
    for o in &operations {
        println!("  {} Type: {}", o.name().unwrap(), o.op_type().unwrap());
    }
    let unused_input_mappings = import_results.missing_unused_input_mappings().unwrap();
    println!("Unused input mappings: {}\n", unused_input_mappings.len());
    for (a, b) in &unused_input_mappings {
        println!("  {} {}", a, b);
    }

    (graph, stride)
}

pub fn image_to_mask(
    graph: &mut Graph,
    session: &mut Session,
    image: &DynamicImage,
    stride: u32,
) -> DynamicImage {
    fn sigmoid(val: f32) -> f32 {
        let neg_val = -val;
        let denominator = 1. + neg_val.exp();
        1. / denominator
    }

    let width = image.width() as usize;
    let height = image.height() as usize;

    // Setup tensor args
    let vec_size = width * height * 3;
    let mut flattened: Vec<f32> = Vec::with_capacity(vec_size);

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32).to_rgb();
            // Resnet preprocessing
            let red = pixel.0[0] as f32 - 123.15;
            let green = pixel.0[1] as f32 - 115.9;
            let blue = pixel.0[2] as f32 - 103.06;

            /*
            // Mobilenet preprocessing
            let red = pixel.0[0] as f32 / 127.5 - 1.;
            let green = pixel.0[1] as f32 / 127.5 - 1.;
            let blue = pixel.0[2] as f32 / 127.5 - 1.;
            */
            flattened.push(red);
            flattened.push(green);
            flattened.push(blue);
        }
    }
    assert_eq!(flattened.len(), vec_size);

    let input = Tensor::new(&[1, height as u64, width as u64, 3])
        .with_values(&flattened)
        .unwrap();

    let mut args: SessionRunArgs = SessionRunArgs::new();
    args.add_feed(
        &graph.operation_by_name_required("sub_2").unwrap(),
        0,
        &input,
    );

    let segments_token = args.request_fetch(
        &graph.operation_by_name_required("float_segments").unwrap(),
        0,
    );

    session.run(&mut args).unwrap();

    let segments_res: Tensor<f32> = args.fetch(segments_token).unwrap();

    let segment_height = segments_res.shape().index(1).unwrap() as usize;
    let segment_width = segments_res.shape().index(2).unwrap() as usize;

    let mut sig_map: Vec<Vec<f32>> = vec![vec![0f32; segment_height]; segment_width];
    for x in 0..segment_width {
        for y in 0..segment_height {
            let seg_val = segments_res.get(&[0, y as u64, x as u64, 0]);
            sig_map[x][y] = sigmoid(seg_val);
        }
    }

    let sig_map = sig_map;
    let half_stride = stride / 2;
    let mean_sig = |x: u32, y: u32| -> f32 {
        let step_x = (x as i32 - half_stride as i32) / stride as i32 - 1;
        let step_y = (y as i32 - half_stride as i32) / stride as i32 - 1;

        assert!(step_x + 1 > -1);
        assert!(step_y + 1 > -1);

        let x1: usize = if step_x < 0 { 0 } else { step_x as usize };
        let x2: usize = (1 + step_x) as usize;

        let y1 = if step_y < 0 { 0 } else { step_y as usize };
        let y2 = (1 + step_y) as usize;

        let x_remainder = if x < half_stride {
            0
        } else {
            (x + half_stride) % stride
        };
        let y_remainder = if y < half_stride {
            0
        } else {
            (y + half_stride) % stride
        };

        let part_x1 = (stride - x_remainder) as f32 / stride as f32;
        let part_x2 = 1. - part_x1;
        let part_y1 = (stride - y_remainder) as f32 / stride as f32;
        let part_y2 = 1. - part_y1;

        let mean_x1 = sig_map[x1][y1] * part_y1 + sig_map[x1][y2] * part_y2;
        let mean_x2 = sig_map[x2][y1] * part_y1 + sig_map[x2][y2] * part_y2;

        let mean = mean_x1 * part_x1 + mean_x2 * part_x2;
        mean
    };

    let output_img = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        let sig_val = mean_sig(x, y);
        let shade_val: u8 = if sig_val >= 0.75 { 255 } else { 0 };

        image::Rgb([shade_val, shade_val, shade_val])
    });

    DynamicImage::ImageRgb8(output_img)
}
