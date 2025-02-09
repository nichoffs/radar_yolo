from argparse import ArgumentParser

from ultralytics import YOLO


def main():
    argparser = ArgumentParser()
    argparser.add_argument("-i", "--sample-idx", default=3000)

    args = argparser.parse_args()

    MODEL_PTH = "/home/nic/code/radar_yolo/runs/obb/train3/weights/best.pt"
    IMG_PTH = f"/mnt/sdc-wdc/radar_net_dataset/images/{args.sample_idx}.png"

    model = YOLO(MODEL_PTH)

    results = model(IMG_PTH)

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        print(type(obb.xyxyxyxy))
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk


if __name__ == "__main__":
    main()
