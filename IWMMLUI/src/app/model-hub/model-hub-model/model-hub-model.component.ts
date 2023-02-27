import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ActivatedRoute } from '@angular/router';
import { IModelHubModel, ModelHubModel } from 'src/app/Models/modelHubModel';

@Component({
  selector: 'app-model-hub-model',
  templateUrl: './model-hub-model.component.html',
  styleUrls: ['./model-hub-model.component.scss']
})
export class ModelHubModelComponent implements OnInit {
  modelHubDetails: IModelHubModel[] = ModelHubModel;
  htmlFilePath: SafeHtml = '';
  modelDownloadPath: string = "";
  modelDownloadName: string = "";
  sourceCodeDownloadPath: string = "";
  sourceCodeDownloadName: string = "";
  datasetDownloadPath: string = "";
  datasetDownloadName: string = "";
  isModelAvailable: boolean = false;
  isDatasetAvailable: boolean = false;
  isSourceCodeAvailable: boolean = false;

  constructor( private activatedRoute: ActivatedRoute,
    private sanitizer: DomSanitizer, private http: HttpClient) { }

  ngOnInit(): void {

    let modelId = this.activatedRoute.snapshot.params['id']
    let modelDetails = this.modelHubDetails.filter(item => item.id == modelId)[0];
  
    

    this.http.get(modelDetails.htmlFilePath, { responseType: 'text' })
      .subscribe(logo => {
        this.htmlFilePath = this.sanitizer.bypassSecurityTrustHtml(logo);
      });
   
  }

}
