# ## install below client library on local before executing code
# ## pip install azure-storage-file-datalake
# import logging
# import os, uuid, sys
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core._match_conditions import MatchConditions
from azure.storage.filedatalake._models import ContentSettings

def ConnectToStorageAcc(StorageAccName, StorageAccKey):
    
    try:  
        global ServiceClient

        ServiceClient = DataLakeServiceClient(account_url="{}://{}.dfs.core.windows.net".format(
            "https", StorageAccName), credential=StorageAccKey)
    
    except Exception as e:
        print(e)
# def ListDirectoryContent(Container,FolderPath):
#     try:
        
#         ContainerClient = ServiceClient.get_file_system_client(file_system=Container)

#         paths = ContainerClient.get_paths(path=FolderPath)

#         for path in paths:
#             print(path.name + '\n')

#     except Exception as e:
#      print(e)

def upload_file_to_directory(SourcePath):
    try:

        file_system_client = ServiceClient.get_file_system_client(file_system="finaldataiwm")

        directory_client = file_system_client.get_directory_client("test")
        
        file_client = directory_client.create_file("Dataset_Bill1.csv")
        local_file = open(SourcePath,'r')

        file_contents = local_file.read()

        file_client.append_data(data=file_contents, offset=0, length=len(file_contents))

        file_client.flush_data(len(file_contents))

    except Exception as e:
      print(e)

# logging.info('Establishing Connection With Storage.......')
# ConnectToStorageAcc('iwmstgacc','871wPYug1CXqXhOomD9hJGou/5ulJhgr9uW7MjS8BEhZLrXUfz4FXr26Ve6G8sQsnDQVTNBWRYIT+ASt3kzLcA==')
# logging.info('Connected Successfully')
# # datasets=ListDirectoryContent('finaldataiwm','test')
# upload_file_to_directory()
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

# def upload_file_to_directory():
#     try:

#         file_system_client = service_client.get_file_system_client(file_system="my-file-system")

#         directory_client = file_system_client.get_directory_client("my-directory")
        
#         file_client = directory_client.create_file("uploaded-file.txt")
#         local_file = open("C:\\file-to-upload.txt",'r')

#         file_contents = local_file.read()

#         file_client.append_data(data=file_contents, offset=0, length=len(file_contents))

#         file_client.flush_data(len(file_contents))

#     except Exception as e:
#       print(e)