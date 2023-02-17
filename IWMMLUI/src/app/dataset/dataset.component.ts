import { Component, OnInit } from '@angular/core';
import { MsalService } from '@azure/msal-angular';
import { IDirectoryItem } from '../Models/iDirectoryItem';
import { FileService } from '../Services/fileService';

import * as XLSX from 'xlsx';
import { HttpClient, HttpParams } from '@angular/common/http';
import { map } from 'rxjs';

export interface APIData {
  type: string;
  size: string;
  created_by: string;
  Modified: string;
}

const TABLE_HEADERS: string[] = [
  'Name',
  'Type',
  'Size',
  'Created by',
  'Modified',
];

@Component({
  selector: 'app-dataset',
  templateUrl: './dataset.component.html',
  styleUrls: ['./dataset.component.scss'],
})
export class DatasetComponent implements OnInit {
  headerSource = TABLE_HEADERS;
  fileName = 'ExcelSheet.xlsx';
  searchText = '';
  apiDataSource: any = [];
  ContainerData: any;
  userdata: any;
  containerName = '';
  path = '/';
  loadspinner: boolean = true;
  isContainerLevel: boolean = true;
  uploadDisabled: boolean = true;

  actionButton: string = "uploadFile";
  selectedFile!: File;

  excelData: string = "";
  excelHeader: any[] = [];
  fileData!: File;

  constructor(
    private msal: MsalService,
    private fileService: FileService,
    private http: HttpClient

  ) { }

  ngOnInit(): void {

    this.GetBlobs();

  }

  // getTableData() {
  //   this.fileService.getTableData().subscribe((response) => {
  //     console.log(response);
  //     this.apiDataSource = [];
  //     response.forEach((x: IDirectoryItem) => {
  //       this.apiDataSource.push(x);
  //     });
  //     console.log(this.apiDataSource);
  //   });
  // }


  GetBlobs() {
    this.fileService.GetBlobs().subscribe(response => {
      console.log(response)
      this.apiDataSource = response
    })
  }


  // uploadFile(event: Event) {
  //   let target = <HTMLInputElement>event.target;
  //   if (!target.files?.length) {
  //     return;
  //   }
  //   const file = target.files[0];

  //   let params = new HttpParams();

  //   params = params.append('container_name', this.containerName);
  //   params = params.append('path', this.path);

  //   const formData = new FormData();
  //   formData.append('file', file);

  //   this.fileService.uploadFile(params, formData).subscribe((result: any) => {
  //     if (result) {
  //       this.apiDataSource.push(result);
  //       alert('uploaded sucessfully!!');
  //     }
  //   });

  // }

  uploadFile(event: Event) {
    let target = <HTMLInputElement>event.target;
    if (!target.files?.length) {
      return;
    }
    const file = target.files[0];
    let params = new HttpParams();
    
   // params = params.append('container_name', this.containerName);
    params = params.append('path', this.path);

    const formData = new FormData();
    formData.append('file', file);
    this.fileService.uploadFile(params, formData).subscribe((result: any) => {
      if (result) {
       
        alert("File uploaded sucessfully.");
      }
    },
      (error) => {
        console.log('Error', error);
        alert('Something went wrong.');
      });

    }

   // http://127.0.0.1:5000/upload_file_to_directory?path=C:\Users\Sivakumar.Kokku\Downloads\Upload.png

  }
