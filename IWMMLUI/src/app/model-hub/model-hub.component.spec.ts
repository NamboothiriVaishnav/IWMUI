import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelHubComponent } from './model-hub.component';

describe('ModelHubComponent', () => {
  let component: ModelHubComponent;
  let fixture: ComponentFixture<ModelHubComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ModelHubComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ModelHubComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
