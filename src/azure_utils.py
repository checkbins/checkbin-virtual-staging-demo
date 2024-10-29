from azure.storage.blob import BlobServiceClient
from io import BytesIO
from config.config import AZURE_ACCOUNT_NAME, AZURE_ACCOUNT_KEY

def get_connection_string():
    return f"DefaultEndpointsProtocol=https;AccountName={AZURE_ACCOUNT_NAME};AccountKey={AZURE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"

def get_container_client(container_name: str):
    connection_string = get_connection_string()
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    return blob_service_client.get_container_client(container_name)

def upload_to_azure(filename, image):
    container_client = get_container_client("virtual-staging-demo")    
    output_buffer = BytesIO()
    image.save(output_buffer, format='PNG')
    output_buffer.seek(0)
    blob_client = container_client.upload_blob(name=filename, data=output_buffer)
    print(f"Uploaded blob URL: {blob_client.url}")
    return blob_client.url