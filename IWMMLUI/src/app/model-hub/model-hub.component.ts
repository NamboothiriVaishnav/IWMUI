import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { IModelHubModel, ModelHubModel } from '../Models/modelHubModel';


@Component({
  selector: 'app-model-hub',
  templateUrl: './model-hub.component.html',
  styleUrls: ['./model-hub.component.scss']
})
export class ModelHubComponent implements OnInit {
  modelHubDetails: IModelHubModel[] = ModelHubModel;
  constructor(private router: Router) { }

  ngOnInit(): void {
  }
  showModel(modelId: any) {    
    this.router.navigate(['/modelhubmodel', modelId]);
  }

}
