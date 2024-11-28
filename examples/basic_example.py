"""
Basic example demonstrating core functionality.

This example shows how to:
1. Create a simple data structure
2. Perform basic operations
3. Display results
"""

def demonstrate_basic_operations():
    # Create a sample data structure
    data = {
        'values': [1, 2, 3, 4, 5],
        'metadata': {
            'description': 'Sample dataset',
            'version': '1.0'
        }
    }
    
    # Perform some operations
    result = {
        'sum': sum(data['values']),
        'average': sum(data['values']) / len(data['values']),
        'metadata': data['metadata']
    }
    
    # Display results
    print("Results:")
    print(f"Sum: {result['sum']}")
    print(f"Average: {result['average']}")
    print(f"Description: {result['metadata']['description']}")

if __name__ == "__main__":
    demonstrate_basic_operations()
