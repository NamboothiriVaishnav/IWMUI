import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { DataModel, IDataModel } from '../Models/dataModel';




@Component({
  selector: 'app-data-model',
  templateUrl: './data-model.component.html',
  styleUrls: ['./data-model.component.scss']
})
export class DataModelComponent implements OnInit {
  dataModelDetails: IDataModel[] = DataModel;
  dataModelInnerHtmlPage: any;

  constructor(private sanitizer: DomSanitizer, private http: HttpClient) { }

  ngOnInit(): void {
  }
  showModel(modelId: any) {
    let dataModelHtmlFileDetail = this.dataModelDetails.filter(x => x.id === modelId)[0].htmlFilePath;
    this.http.get(dataModelHtmlFileDetail, { responseType: 'text' })
      .subscribe(logo => {
        this.dataModelInnerHtmlPage = this.sanitizer.bypassSecurityTrustHtml(logo);
      }); 
  }

}
