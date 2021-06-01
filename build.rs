use pyo3::prelude::*;
use serde_json::Value;
use std::io::Cursor;
use std::path::{Path, PathBuf};

const BASE_PATH: &str = "https://storage.googleapis.com/tfjs-models/savedmodel";
const BASE_TMP: &str = "./target/tmp";

#[tokio::main]
async fn main() {
    // TODO: Re-run if models are missing or are older than two weeks?

    // Import python libraries
    let gil = Python::acquire_gil();
    let convert_graph = import_graph_convert(&gil);

    // Create assets dir
    let output_path = Path::new("./assets/models/");
    create_dir_if_not_exists(output_path);

    let model_sets = get_model_set_definitions();
    for set in &model_sets {
        // Create temp dir
        let model_name = set.0.to_string().replace('/', "_");
        let temp_dir_name = format!("{}/{}", &BASE_TMP, &model_name);
        let temp_model_path = Path::new(temp_dir_name.as_str());
        create_dir_if_not_exists(temp_model_path);

        // Download models
        fetch_model_set(&set, temp_model_path).await;

        // Convert models
        let model_files = &set.1;
        for model_file in model_files {
            let stride = model_file.split_at(5).1; // Remove "model"
            let stride = stride.split_at(stride.len() - 5).0; // Remove ".json"

            let output_path = output_path.join(format!("{}{}.pb", model_name, stride).as_str());
            let input_path = temp_model_path.join(model_file);
            if !input_path.exists() {
                panic!("Input path {:?} missing!", input_path);
            }
            delete_if_exists(&output_path);

            let input = input_path.into_os_string();
            let input = input.into_string().unwrap();
            let input = input.as_str();
            let output = output_path.into_os_string();
            let output = output.into_string().unwrap();
            let output = output.as_str();

            convert_graph
                .call1((vec![input, output, "-s"],))
                .unwrap_or_else(|err| {
                    panic!("Error converting to frozen graph: {}", err);
                });
        }
    }

    // Cleanup
    std::fs::remove_dir_all(BASE_TMP).unwrap_or_else(|err| {
        panic!(
            "Failed to delete \"{}\" after building assets: {}",
            BASE_TMP, err
        );
    });
}

async fn fetch_shard(shard: &str, base_url: &String, path: &Path) {
    let mut url = base_url.clone();
    url.push_str(shard);
    let output_path = path.join(shard);

    if !output_path.exists() {
        let mut file = std::fs::File::create(output_path).unwrap_or_else(|err| {
            panic!("{}", err);
        });
        let response = reqwest::get(url).await.unwrap_or_else(|err| {
            panic!("{}", err);
        });
        let mut content = Cursor::new(response.bytes().await.unwrap_or_else(|err| {
            panic!("{}", err);
        }));
        std::io::copy(&mut content, &mut file).unwrap_or_else(|err| {
            panic!("{}", err);
        });
        file.sync_all().unwrap_or_else(|err| {
            panic!("{}", err);
        });
    }
}

async fn fetch_model_set(set: &(&str, Vec<&str>), path: &Path) {
    let (url_path_frag, model_name_set) = set;
    let base_url = format!("{}/{}/", BASE_PATH, url_path_frag);
    let mut json_data: Vec<bytes::Bytes> = vec![];

    for model_name in model_name_set {
        let mut url = base_url.clone();
        url.push_str(model_name);
        let output_path = path.join(model_name);

        let response = reqwest::get(url).await.unwrap_or_else(|err| {
            panic!("{}", err);
        });
        let response_data = response.bytes().await.unwrap_or_else(|err| {
            panic!("{}", err);
        });
        json_data.push(response_data.clone());

        if !output_path.exists() {
            let mut file = std::fs::File::create(output_path).unwrap_or_else(|err| {
                panic!("File could not be created: {}", err);
            });
            let mut content = Cursor::new(response_data);
            std::io::copy(&mut content, &mut file).unwrap_or_else(|err| {
                panic!("{}", err);
            });
            file.sync_all().unwrap_or_else(|err| {
                panic!("{}", err);
            });
        }
    }

    // Parse models
    let json: Value = serde_json::from_slice(json_data[0].as_ref()).unwrap();
    let weights_manifest = json.get("weightsManifest").unwrap().as_array().unwrap();
    let shards = weights_manifest[0]
        .get("paths")
        .unwrap()
        .as_array()
        .unwrap();

    // Download shards
    for shard in shards {
        let shard = shard.as_str().unwrap();
        fetch_shard(shard, &base_url, &path).await;
    }
}

fn create_dir_if_not_exists(path: &Path) {
    match std::fs::create_dir_all(path) {
        Ok(_) => {}
        Err(e) => {
            use std::io::ErrorKind::AlreadyExists;
            if e.kind() != AlreadyExists {
                panic!("{}", e);
            }
        }
    }
}

fn get_model_set_definitions() -> Vec<(&'static str, Vec<&'static str>)> {
    let resnet_models: (&str, Vec<&str>) = (
        "bodypix/resnet50/float",
        vec!["model-stride16.json", "model-stride32.json"],
    );
    let mobilenet_050_models: (&str, Vec<&str>) = (
        "bodypix/mobilenet/float/050",
        vec!["model-stride8.json", "model-stride16.json"],
    );
    let mobilenet_075_models: (&str, Vec<&str>) = (
        "bodypix/mobilenet/float/075",
        vec!["model-stride8.json", "model-stride16.json"],
    );
    let mobilenet_100_models: (&str, Vec<&str>) = (
        "bodypix/mobilenet/float/100",
        vec!["model-stride8.json", "model-stride16.json"],
    );
    vec![
        resnet_models,
        mobilenet_050_models,
        mobilenet_075_models,
        mobilenet_100_models,
    ]
}

fn import_graph_convert(gil: &GILGuard) -> &PyAny {
    let py = gil.python();
    let tfjs_graph_converter = PyModule::import(py, "tfjs_graph_converter.converter").unwrap_or_else(|err|{
        panic!("Missing required dependencies.\nMake sure you have tfjs_graph_converter installed for python3.\n{}", err);
    });

    let convert_func = tfjs_graph_converter
        .getattr("convert")
        .unwrap_or_else(|err| {
            panic!(
                "Error importing \"convert\" from tfjs_graph_converter python3 module: {}",
                err
            );
        });

    convert_func
}

fn delete_if_exists(path: &PathBuf) {
    if path.exists() {
        std::fs::remove_file(path).unwrap_or_else(|err| {
            let path = path.clone().into_os_string().into_string().unwrap();
            panic!("Failed to delete \"{}\": {}", path, err);
        });
    }
}
