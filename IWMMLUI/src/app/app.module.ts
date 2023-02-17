import { NgModule, CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomeComponent } from './home/home.component';
import { AutoMlComponent } from './auto-ml/auto-ml.component';
import { ModelHubComponent } from './model-hub/model-hub.component';
import { HeaderComponent } from './Shared/header/header.component';
import { MsalGuard, MsalInterceptor, MsalBroadcastService, MsalModule, MsalService, MsalRedirectComponent } from '@azure/msal-angular';
import { BrowserCacheLocation, InteractionType, PublicClientApplication } from '@azure/msal-browser';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { NgxSpinnerModule } from 'ngx-spinner';
import { ToastrModule } from 'ngx-toastr';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { environment } from 'src/environments/environment';
import { DataModelComponent } from './data-model/data-model.component';
import { DatasetComponent } from './dataset/dataset.component';
import { ModelHubModelComponent } from './model-hub/model-hub-model/model-hub-model.component';

import { FormsModule, ReactiveFormsModule } from '@angular/forms';


const environmentVariable = environment;
@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    AutoMlComponent,
    ModelHubComponent,
    HeaderComponent,
    DataModelComponent,
    DatasetComponent,
    ModelHubModelComponent,
   
   
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    ReactiveFormsModule,
    FormsModule,
    MsalModule.forRoot(new PublicClientApplication({ // MSAL Configuration
      auth: {
        clientId: environmentVariable.clientId,
        authority: environmentVariable.authority,
        redirectUri: environmentVariable.redirectUrl
      },
      cache: {
        cacheLocation: BrowserCacheLocation.LocalStorage,
        storeAuthStateInCookie: true, // set to true for IE 11
      },
      system: {
        loggerOptions: {
          loggerCallback: () => { },
          piiLoggingEnabled: false
        }
      }
    }), {
      interactionType: InteractionType.Redirect, // MSAL Guard Configuration
    }, {
      interactionType: InteractionType.Redirect,
      protectedResourceMap: new Map([
        ['https://graph.microsoft.com/v1.0/me', ['user.read']],
        ['http://localhost:4200/', null]
      ])
    }),
    HttpClientModule,
    ToastrModule.forRoot(),
    NgxSpinnerModule,
    BrowserAnimationsModule,
  ],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  providers: [
    {
      provide: HTTP_INTERCEPTORS,
      useClass: MsalInterceptor,
      multi: true
    },
    MsalService,
    MsalGuard,
    MsalBroadcastService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
