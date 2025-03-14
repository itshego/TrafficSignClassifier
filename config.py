import os

# Color thresholds for traffic sign detection
# HSV color space values for blue and red colors
LOWER_BLUE = [105, 105, 55]  # Minimum HSV values for navy blue
UPPER_BLUE = [140, 255, 165]  # Maximum HSV values for navy blue
LOWER_RED1 = [0, 100, 100]    # Minimum HSV values for red (first range)
UPPER_RED1 = [10, 255, 255]   # Maximum HSV values for red (first range)
LOWER_RED2 = [160, 100, 100]  # Minimum HSV values for red (second range)
UPPER_RED2 = [180, 255, 255]  # Maximum HSV values for red (second range)

# Image processing parameters for sign detection
MIN_CONTOUR_AREA = 100  # Minimum area for a valid contour
ELLIPSE_SIZE_THRESHOLD = 0.3  # Threshold for ellipse size validation
KERNEL_SIZE = (3, 3)  # Kernel size for morphological operations

# File paths for input/output operations
DESKTOP_LOCATION = os.path.join(os.path.expanduser("~"), "Desktop")
DEFAULT_FOLDER_PATH = os.path.join(DESKTOP_LOCATION, "Hackathon", "kirpilmis_resimler_2024-07-17-1")
OUTPUT_FOLDER = "output"  # Folder for storing results
OUTPUT_FILE_NAME = "results.txt"  # Name of the results file
DEBUG_FOLDER = "debug_images"  # Folder for storing debug images

# Supported image file extensions
SUPPORTED_EXTENSIONS = ('.jpeg', '.jpg', '.png')

# Scoring system for different detection outcomes
# Positive scores for correct detections, negative for errors
SCORING = {
    'invalid_ellipse': -0.25,  # Invalid ellipse detection
    'invalid_sign': -0.25,     # Invalid sign detection
    'no_sign_detected': -0.25, # No sign found in image
    'left': 0.1,              # Correct left direction detection
    'right': -0.5,            # Correct right direction detection
    'image_read_error': -1.0   # Error reading image file
}

# Text constants for region identification
LEFT_TOP = 'Left_Top'
RIGHT_TOP = 'Right_Top'
LEFT_BOTTOM = 'Left_Bottom'
RIGHT_BOTTOM = 'Right_Bottom'
LEFT = 'LEFT'
RIGHT = 'RIGHT'

# Error messages for different failure cases
ERRORS = {
    'image_read_error': 'Image could not be read',
    'no_sign_detected': 'No sign detected',
    'invalid_sign': 'No valid sign found',
    'invalid_ellipse': 'Invalid ellipse detection:',
    'invalid_sign_name': 'Error in sign name.'
}

# Output messages for different operations
OUTPUT = {
    'debug_images_saved': 'Debug images saved to {} folder.',
    'processing_completed': 'Processing completed. Results and statistics saved to {}',
    'image_read_error': 'Image read error ({}): {}',
    'region_stats': {
        'total_pixels': 'Total pixels: {}',
        'blue_pixels': 'Blue pixels: {}',
        'blue_percentage': 'Blue pixel percentage: {:.2f}%',
        'region': '{} region:'
    },
    'stats': {
        'title': '--- Statistics ---',
        'total_processed': 'Total Processed Images: {}',
        'total_score': 'Total Score: {}',
        'results_title': '--- Results ---',
        'score_format': '{}: {} (Score: {})'
    },
    'progress': {
        'processing_files': 'Processing files'
    }
} 