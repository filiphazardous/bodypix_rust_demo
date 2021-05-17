use std::fs::File;
use std::io::Read;
use std::ops::Index;
use std::path::PathBuf;

use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionRunArgs, Tensor};

use nannou::image;
use nannou::image::{DynamicImage, GenericImageView, Pixel};

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
) -> Vec<Vec<f32>> {
    fn sigmoid(val: f32) -> f32 {
        let neg_val = -val;
        let denominator = 1. + neg_val.exp();
        1. / denominator
    }

    let img_width = image.width();
    let img_height = image.height();

    let target_width = img_width + 1;
    let target_height = img_height + 1;

    let input_image = image.resize_exact(target_width, target_height, image::imageops::Nearest);

    let target_width = input_image.width();
    let target_height = input_image.height();

    println!(
        "{}x{}: {}x{}",
        img_height, img_width, target_height, target_width
    );

    // Setup tensor args
    let vec_size = target_width * target_height * 3;
    let mut flattened: Vec<f32> = Vec::with_capacity(vec_size as usize);

    for y in 0..target_height {
        for x in 0..target_width {
            let pixel = input_image.get_pixel(x as u32, y as u32).to_rgb();
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
    assert_eq!(flattened.len(), vec_size as usize);

    let input = Tensor::new(&[1, target_height as u64, target_width as u64, 3])
        .with_values(&flattened)
        .unwrap();

    println!("Input Image Shape in hwc: {}", input.shape());
    let width_resolution = (input.shape()[1].unwrap() - 1) as u32 / stride + 1;
    let height_resolution = (input.shape()[0].unwrap() - 1) as u32 / stride + 1;
    println!("Resolution: {} {}", width_resolution, height_resolution);

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

    let blocky_sig = |x: usize, y: usize| -> f32 {
        let step_x = 1 + x / stride as usize;
        let step_y = 1 + y / stride as usize;
        sig_map[step_x][step_y]
    };

    let mean_sig = |x: usize, y: usize| -> f32 {
        let step_x = x / stride as usize;
        let step_y = y / stride as usize;

        let x1: usize = step_x;
        let x2: usize = 1 + step_x;
        assert!(x2 < segment_width);

        let y1 = step_y;
        let y2 = 1 + step_y;
        assert!(y2 < segment_height);

        let x_remainder = x as u32 % stride;
        let y_remainder = y as u32 % stride;

        let part_x1 = (stride - x_remainder) as f32 / stride as f32;
        let part_x2 = 1. - part_x1;
        let part_y1 = (stride - y_remainder) as f32 / stride as f32;
        let part_y2 = 1. - part_y1;

        let mean_x1 = sig_map[x1][y1] * part_y1 + sig_map[x1][y2] * part_y2;
        let mean_x2 = sig_map[x2][y1] * part_y1 + sig_map[x2][y2] * part_y2;

        let mean = mean_x1 * part_x1 + mean_x2 * part_x2;

        mean
    };

    let mut mask: Vec<Vec<f32>> = vec![vec![0.; img_height as usize]; img_width as usize];
    for x in 0..img_width {
        for y in 0..img_height {
            let sig_val = mean_sig(x as usize, y as usize);
            mask[x as usize][y as usize] = sig_val;
        }
    }

    mask
}
