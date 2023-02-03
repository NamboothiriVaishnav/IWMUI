# ## install below client library on local before executing code
# ## pip install azure-storage-file-datalake
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core._match_conditions import MatchConditions
from azure.storage.filedatalake._models import ContentSettings
from azure.storage.blob import BlobServiceClient

connection_str="DefaultEndpointsProtocol=https;AccountName=iwmstgacc;AccountKey=871wPYug1CXqXhOomD9hJGou/5ulJhgr9uW7MjS8BEhZLrXUfz4FXr26Ve6G8sQsnDQVTNBWRYIT+ASt3kzLcA==;EndpointSuffix=core.windows.net"
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.storage.filedatalake import DataLakeServiceClient, DataLakeFileClient
import re
import datetime
def listing_blobs_directories(container_name, path):
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    container_client=blob_service_client.get_container_client(container_name)
    blob_list = container_client.walk_blobs(name_starts_with=path)
    file_list=[]
    for blob in blob_list:
        blob_ = BlobClient.from_connection_string(conn_str = connection_str, container_name = container_name, blob_name = blob.name)
        file_dict=blob_.get_blob_properties()["metadata"]
        print(file_dict)
        try:
           del file_dict['hdi_isfolder']
        except:
            pass
        if "/" in blob.name:
            pattern = r"/$"
            match = re.search(pattern, blob.name)
            if match:
                file_dict["name"] = blob.name.split("/")[-2]
            else:
                file_dict["name"] = blob.name.split("/")[-1]
        else:
            file_dict["name"] = blob.name
        file_list.append(file_dict)
    return file_list


connection_string ="DefaultEndpointsProtocol=https;AccountName=iwmstgacc;AccountKey=HgIrVrEcgnhrs0TgXBDyG4g4+22ejK9dqk9l29rc/HegtcGCUbRNUmMGn4M4v6bpuGbLF3XzFnNh+AStKTXChg==;EndpointSuffix=core.windows.net"
container_name = "finaldataiwm/test/"
def upload_file_to_directory(file_path):
    try:
        position=file_path.rfind('\\')
        file_name=file_path[(position+1):]
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        with open(file_path, 'rb') as data:
            blob_client.upload_blob(data)
    except Exception as e:
      print(e)
	