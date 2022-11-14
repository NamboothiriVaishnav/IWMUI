import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomeComponent } from './home/home/home.component';
import { AutoMlComponent } from './auto-ml/auto-ml.component';
import { ModelHubComponent } from './model-hub/model-hub.component';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    AutoMlComponent,
    ModelHubComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
