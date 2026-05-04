"""
Replaces class IDs of YOLO labels with a set class for consistency across merged datasets

"""

from pathlib import Path

def replace_yolo_class_ids(labels_dir, new_class_id):
    labels_dir = Path(labels_dir)

    for label_path in labels_dir.glob("*.txt"):
        new_lines = []

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                parts = line.split()

                # YOLO format should be:
                # class_id x_center y_center width height
                if len(parts) < 5:
                    print(f"Skipping invalid line in {label_path}: {line}")
                    new_lines.append(line)
                    continue

                parts[0] = str(new_class_id)
                new_lines.append(" ".join(parts))

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

        print(f"Updated {label_path} to label class {new_class_id}")


if __name__ == "__main__":
    # labels_directory = "detect_dataset/deer/labels"
    # labels_directory = "detect_dataset/bear/labels"
    # labels_directory = "detect_dataset/bison/labels"
    # labels_directory = "detect_dataset/fox/labels"
    # labels_directory = "detect_dataset/moose/labels"
    # labels_directory = "detect_dataset/wolf/labels"
    # labels_directory = "tmp_labels/"
    labels_directory = "detect_dataset/squirrel/labels"
    class_id_to_use = 6

    replace_yolo_class_ids(labels_directory, class_id_to_use)