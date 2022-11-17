import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { MsalService } from '@azure/msal-angular';
import { forkJoin, Observable } from 'rxjs';
import { IGraphUser } from 'src/app/Models/graphUser';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.scss']
})
export class HeaderComponent implements OnInit {
  userPhoto: string = "";
  defaultPhoto: string = "";
  user!: IGraphUser;
  userName: string = "";
  graphUrl = environment.graphUrl

  constructor(private msalService: MsalService, private http: HttpClient) { }

  ngOnInit(): void {
    this.setAccountInfo();
    this.getUser();

  }

  logout() {
    this.msalService.logout();
  }


  private setAccountInfo(): void {
    let activeAccount = this.msalService.instance.getActiveAccount();
    const accounts = this.msalService.instance.getAllAccounts();
    if (!activeAccount && accounts.length > 0) {
      activeAccount = accounts[0];
      this.msalService.instance.setActiveAccount(activeAccount);
    }
    this.userName = this.getUserName(activeAccount);
  }

  private getUser(): void {
    this.getUserProfileInfo().subscribe(({ userInfo, profilePhoto }) => {
      this.user = userInfo;
      if (profilePhoto) {
        this.createImageFromBlob(profilePhoto);
      }
    });
    if (!this.userPhoto) {
      const userName = this.userName;
      this.defaultPhoto = userName
        .split(' ')
        .map(name => name[0])
        .join('')
        .toUpperCase();
    }
  }

  private getUserProfileInfo(): Observable<{ userInfo: IGraphUser; profilePhoto: Blob }> {
    return forkJoin({
      userInfo: this.getUserInfo(),
      profilePhoto: this.getUserProfilePhoto()
    });
  }

  private createImageFromBlob(image: Blob): void {
    const reader = new FileReader();
    reader.addEventListener('load', () => {
      this.userPhoto = reader.result as string;
    });

    if (image) {
      reader.readAsDataURL(image);
    }
  }

  private getUserInfo(): Observable<IGraphUser> {
    const url = `${this.graphUrl}/me`;
    return this.http.get<IGraphUser>(url);
  }

  private getUserProfilePhoto(): Observable<any> {
    const url = `${this.graphUrl}/me/photo/$value`;
    return this.http.get(url, {
      headers: new HttpHeaders({
        Accept: 'image/jpeg'
      }),
      responseType: 'blob'
    });
  }

  private getUserName(account: any): string {
    return (account && account.username) || 'N/A';
  }

}
