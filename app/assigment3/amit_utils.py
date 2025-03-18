import json
from roboflow import Roboflow
rf = Roboflow(api_key="WkCNtz2v2MkWw0TNp483")
project = rf.workspace("gl-ccyuo").project("b-xsr35")
version = project.version(1)
dataset = version.download("coco")

def merge_coco_annotations(ann_file1, ann_file2, output_file):
    # Load annotation files
    with open(ann_file1, 'r') as f:
        data1 = json.load(f)
    with open(ann_file2, 'r') as f:
        data2 = json.load(f)

    # Since there is no overlap of categories, we can simply merge them:
    categories = data1['categories'] + data2['categories']

    # Merge images and annotations by concatenation.
    images = data1['images'] + data2['images']
    annotations = data1['annotations'] + data2['annotations']

    # Create the merged COCO dictionary
    merged_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Write the merged annotations to a file
    print(f"write to output file: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

def update_face_annotation(input_path, output_path,id_offset,ann_offset):
    """
    Reads a COCO annotation file from input_path, changes all annotations with category_id 1 to 7,
    and writes the updated annotations to output_path.
    """
    # Load the COCO annotation file.
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Update annotation labels: change category_id 1 to 7.
    for ann in data.get("annotations", []):
        if ann.get("category_id") == 1:
            ann["category_id"] = 2
        ann['image_id'] = ann.get('image_id')+id_offset
        ann['id'] = ann['id']+ann_offset

    # Optionally, update the categories list if needed.
    # For example, if there's a category with id 1, you might want to change it to 7:
    new_cat = []
    for cat in data.get("categories", []):
        if cat.get("id") == 1:
            cat["id"] = 2
            new_cat.append(cat)

    for img in data.get("images",[]):
        img["id"] = img["id"]+id_offset
    # Write the updated data to the output file.
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_offsets(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
        image_id_offset = data['images'][-1]['id']
        annotation_id_offset = data['annotations'][-1]['id']
        return image_id_offset,annotation_id_offset


def unit_annon_files(ann_file1, ann_file2,output_file):
    image_id_offset, annotation_id_offset = get_offsets(ann_file2)
    new_ann_file2 = ann_file2.replace(".json","_new.json")
    update_face_annotation(ann_file2, new_ann_file2, image_id_offset,annotation_id_offset)
    merge_coco_annotations(new_ann_file2,ann_file1,output_file)



# # Example usage:
banana = "b-1/train/_annotations.coco.json"
face = "app/assigment3/Face Detection.v25i.coco/train/_annotations.coco.json"
face_out = "app/assigment3/Face Detection.v25i.coco/train/_annotations2.coco.json"
# face_out_eyes = "app/assigment3/Face Detection.v25i.coco/train/_annotations3.coco.json"
output = "app/assigment3/banana_face_annotations.coco.json"
# unit_annon_files(face,banana,output)
# output_eyes = "app/assigment3/_annotations.coco.json"
# eyes = "app/assigment3/Eyes_detection.v1i.coco/train/_annotations.coco.json"
# update_face_annotation(face, face_out,2374,3039)
# merge_coco_annotations(banana, face_out, output)
# exit(0)
# exit(0)
import json
import numpy as np
from sklearn.cluster import KMeans

def compute_global_anchors(ann_file, num_clusters=4, default_image_size=(640, 480)):
    """
    Reads a COCO annotation file and computes global anchors based on normalized
    bbox widths and heights, using k-means clustering on all annotations.

    Args:
        ann_file (str): Path to the COCO annotation JSON file.
        num_clusters (int): Number of clusters (anchors) to compute.
        default_image_size (tuple): (width, height) for normalization if an image size isn't provided.

    Returns:
        numpy.ndarray: Array of shape [num_clusters, 2] containing normalized anchors.
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # Build a lookup for image sizes, if available.
    image_sizes = {}
    for img in data.get('images', []):
        image_sizes[img['id']] = (
            img.get('width', default_image_size[0]),
            img.get('height', default_image_size[1])
        )

    # Collect all normalized [width, height] pairs for bounding boxes.
    bbox_dimensions = []
    for ann in data.get('annotations', []):
        image_id = ann['image_id']
        bbox = ann['bbox']  # Format: [x, y, width, height]
        img_w, img_h = image_sizes.get(image_id, default_image_size)
        norm_width = bbox[2] / img_w
        norm_height = bbox[3] / img_h
        bbox_dimensions.append([norm_width, norm_height])

    bbox_dimensions = np.array(bbox_dimensions)
    clusters = min(len(bbox_dimensions), num_clusters)
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(bbox_dimensions)
    anchors = kmeans.cluster_centers_
    return anchors
import json

# # Path to the annotation file
# ann_file = "app/assigment3/Chess-Pieces-Detection-1/train/_annotations.coco.json"

# # Load the JSON file and extract categories
# with open(ann_file, "r") as f:
#     data = json.load(f)
# print(data["categories"])
# # Extract unique category IDs and names
# categories = {cat["id"]: cat["name"] for cat in data}
# print(categories)
# ann_file = "app/assigment3/Face Detection.v25i.coco/train/_annotations2.coco.json"
anchors = compute_global_anchors(banana, num_clusters=12, default_image_size=(640, 640))
print("Global Anchors (normalized width, height):")
print(anchors)
