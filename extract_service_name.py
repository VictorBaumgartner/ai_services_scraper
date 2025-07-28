import json

def extract_service_names(input_file):
    """
    Extracts all Service_Name values from a JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        
    Returns:
        list: List of all Service_Name values
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        service_names = []
        
        # Handle both list of services or single service object
        if isinstance(data, list):
            for item in data:
                if "Service_Name" in item:
                    service_names.append(item["Service_Name"])
        elif isinstance(data, dict) and "Service_Name" in data:
            service_names.append(data["Service_Name"])
        
        return service_names
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

# Example usage:
service_names = extract_service_names('ai_services_comprehensive.json')
print("List of Service Names:")
for name in service_names:
    print(f"- {name}")