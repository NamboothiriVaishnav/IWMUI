import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { MsalService } from '@azure/msal-angular';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root',
})
export class FileService {
  // tableData: string =
  //   '';
  // newContainerUrl: string = '';
  // folderContents: string = ' ';
  uploadFileURL: string = "http://127.0.0.1:5000/upload_file_to_directory";
  

  constructor(private http: HttpClient, private msalService: MsalService) { }

  // getTableData(): Observable<any> {
  //   return this.http.get<any>(this.tableData);
  // }

  // getFolderContents(containerName: string, path: string): Observable<any> {
  //   const params = new HttpParams()
  //     .append('container_name', containerName)
  //     .append('path', path);
  //   return this.http.get<any>(this.folderContents, {
  //     params: params,
  //   });
  // }

  // createContainer(containerName: string) {
  //   var accountInfo = this.msalService.instance.getAllAccounts()[0];
  //   console.log(accountInfo);
  //   const params = new HttpParams()
  //     .append('user_email', accountInfo.username)
  //     .append('container_name', containerName);
  //   return this.http.post<any>(this.newContainerUrl, '', {
  //     params: params,
  //   });
  // }

  // createDirectory(directoryName: string) {
  //   var accountInfo = this.msalService.instance.getAllAccounts()[0];
  //   const params = new HttpParams()
  //     .append('user_email', accountInfo.username)
  //     .append('folder_name', directoryName);
  //   return this.http.post<any>(this.newContainerUrl, '', {
  //     params: params,
  //   });
  // }


  GetBlobs() {
    return this.http.get<any>(environment.apiUrl + 'listing_blobs_directories');
  }

  uploadFile(params: HttpParams, formData: any) {
    let accountInfo = this.msalService.instance.getAllAccounts()[0];
    let header = new HttpHeaders();
   // params = params.append('user_email', accountInfo.username);
   // params = params.append('path', path);
    // return this.http.post<any>(this.uploadFileURL, formData, {
    //   params: params,
    //   headers: header,
     
    // });
    return this.http.post<any>(environment.apiUrl + 'upload_file_to_directory', formData, {
      params: params,
      headers: header,  
    });
    
  }


  


}


