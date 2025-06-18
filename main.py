from PIL import Image

def analyze_image(image_path):
    image_results=[]
    try:
        with Image.open(image_path) as img:
            
            width, height = img.size
            analysis_result = {
                'width': width,
                'height': height,
                'mode': img.mode,
                'format': img.format,
                'size': img.size,
                'info': img.info
            }
    except Exception as e:
        print(f"Error analyzing image: {e}")
        analysis_result = {'error': str(e)}
    image_results.append(analysis_result)
    return image_results

print(analyze_image('Data/test/0dd71fdd2db2d0beb278eddb5692a156fb79a40c.temp.jpeg')) 
