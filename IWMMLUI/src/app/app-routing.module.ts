import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { MsalGuard } from '@azure/msal-angular';
import { ModelHubComponent } from './model-hub/model-hub.component';
import { AutoMlComponent } from './auto-ml/auto-ml.component';
import { DataModelComponent } from './data-model/data-model.component';
import { DatasetComponent } from './dataset/dataset.component';
import { ModelHubModelComponent } from './model-hub/model-hub-model/model-hub-model.component';

const routes: Routes = [

  { path: '', component: HomeComponent, canActivate: [MsalGuard] },
  { path: 'home', component: HomeComponent, canActivate: [MsalGuard] },
  { path: 'modelhub', component: ModelHubComponent, canActivate: [MsalGuard] },
  { path: 'automl', component: AutoMlComponent, canActivate: [MsalGuard] },
  { path: 'datamodel', component: DataModelComponent, canActivate: [MsalGuard] },
  { path: 'dataset', component: DatasetComponent, canActivate: [MsalGuard] },
  { path: 'modelhubmodel/:id', component: ModelHubModelComponent, canActivate: [MsalGuard] }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
