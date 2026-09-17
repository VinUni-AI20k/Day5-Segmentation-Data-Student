import json
import argparse

def rgb_to_hex(rgb):
    """Converts an RGB list [R, G, B] to a Hex color string."""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def convert_to_cvat_raw(input_filename, output_filename):
    # Load the input JSON data
    try:
        with open(input_filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        return

    cvat_labels = []
    colors_dict = data.get("colors", {})

    # Iterate through the classes and build the CVAT format
    for class_name in data.get("classes", []):
        # Convert the RGB array to a Hex string, defaulting to black if missing
        rgb_color = colors_dict.get(class_name, [0, 0, 0])
        hex_color = rgb_to_hex(rgb_color)

        cvat_label = {
            "name": class_name,
            "color": hex_color,
            "attributes": [],
            "type": "any"
        }
        cvat_labels.append(cvat_label)

    # Save to the output JSON file
    with open(output_filename, 'w') as f:
        json.dump(cvat_labels, f, indent=4)
    
    print(f"Successfully converted labels and saved to '{output_filename}'.")

if __name__ == "__main__":
    # Set up argument parsing for command line usage
    parser = argparse.ArgumentParser(description="Convert custom classes JSON to CVAT Raw format.")
    
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        default="classes.json", 
        help="Path to the input JSON file (default: classes.json)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="cvat_raw_labels.json", 
        help="Path to the target JSON file (default: cvat_raw_labels.json)"
    )
    
    args = parser.parse_args()
    
    # Run the conversion
    convert_to_cvat_raw(args.input, args.output)