import json

def remove_keys_from_json(input_file, output_file, keys_to_remove):
    """
    Removes specified keys from a JSON file and saves the modified version.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the modified JSON
        keys_to_remove (list): List of keys to remove from the JSON
    """
    try:
        # Explicitly specify UTF-8 encoding when opening the file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process the data (works for both list of dicts or single dict)
        if isinstance(data, list):
            for item in data:
                for key in keys_to_remove:
                    item.pop(key, None)
        elif isinstance(data, dict):
            for key in keys_to_remove:
                data.pop(key, None)
        
        # Also specify UTF-8 encoding when writing the file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully removed keys and saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
keys_to_remove = [
    "Carbon_Footprint", 
    "API_Features", 
    "Performance_Metrics", 
    "Category_Group"
]

remove_keys_from_json(
    input_file='ai_services_comprehensive.json',
    output_file='output_ai_services.json',
    keys_to_remove=keys_to_remove
)