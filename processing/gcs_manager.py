from __future__ import annotations
from typing import List

from google.cloud import storage
import google.auth
from google.oauth2 import service_account
import os

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# from .path_manager import InputPathManager, OutputPathManager

class GCSManager:
    """
    Responsible for reading files from Google Cloud Storage
    provided the necessary credentials.
    By default credentials are expected to reside in the root folder
    with the name "log_writer.json". Service account with these credentials
    must have Cloud Storage Viewer, Cloud Storage User, Cloud Storage Admin IAM permission.

    More about generating credentials for service accounts:
    https://cloud.google.com/iam/docs/keys-create-delete
    """
    __client = storage.Client()
    __bucket_name = ""
    __default_credential_file ="./log_writer.json"

    @classmethod
    def authenticate(cls, credentials_file:str="")->None:
        """
        Function that authenticates the account the provided credentials
        belong to. Account with the provided credentials must have
        Cloud Storage Viewer IAM permission to read files from GCS.

        :param: credentials_file        path to .json file containing account's credentials, str
                                        default empty string, in such case credentials are expected 
                                        to reside in the root folder with the name "gcs_reader.json"
        return:
        """
        _cf = credentials_file
        if not credentials_file:
            _cf = cls.__default_credential_file
        credentials:google.auth.credentials.Credentials = service_account.Credentials.from_service_account_file(_cf)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _cf
        cls.__client = storage.Client(credentials=credentials)
    
    @classmethod
    def list_blobs(cls, prefix: str = "", bucket_name:str="") -> List[str]:

        _bucket = cls.__client.bucket(bucket_name)
        return [b.name for b in list(_bucket.list_blobs(prefix=prefix))]
    
    @classmethod
    def delete_blob(cls, fname: str = "", bucket_name:str="") -> List[str]:


        blobs = cls.__client.bucket(bucket_name).list_blobs(prefix=fname)
        fnames = []
        for blob in blobs:
            fnames.append(blob.name)
            blob.delete()
        return fnames
    
    @classmethod
    def load(cls, fname:str, bucket_name:str="")->str:
        """
        Function that loads file contents from the bucket as text into memory.
        Perform authentication before loading the file.

        :param: fname           name of the file to read without the bucket name, str
        :param: bucket_name     name of the bucket the file resides in, str
                                defaults to the config bucket if the param is not 
                                provided
        return: result          file contents, str

        """
        _bucket_name = bucket_name
        if not bucket_name:
            _bucket_name = cls.__bucket_name
        _bucket = cls.__client.bucket(_bucket_name)
        _blob = _bucket.blob(fname)
        return _blob.download_as_text()
    
    @classmethod
    def load_to_local(cls, fname:str, target:str, bucket_name:str="")->bool:
        """
        Function that loads file contents from the bucket as text into memory.
        Perform authentication before loading the file.

        :param: fname           name of the file to read without the bucket name, str
        :param: target          path to the target file, str
        return: result          file contents, str

        """
        _bucket_name = bucket_name
        if not bucket_name:
            _bucket_name = cls.__bucket_name
        _bucket = cls.__client.bucket(_bucket_name)
        _blob = _bucket.blob(fname)
        _blob.download_to_filename(target)
        return True
    
    @classmethod
    def upload(cls, contents:str, 
               fname:str="config.json", 
               bucket_name:str="")->str:
        
        _bucket_name = bucket_name
        if not bucket_name:
            _bucket_name = cls.__bucket_name
        _bucket = cls.__client.bucket(_bucket_name)
        blob = _bucket.blob(fname)

        blob.upload_from_string(contents)
    
    

