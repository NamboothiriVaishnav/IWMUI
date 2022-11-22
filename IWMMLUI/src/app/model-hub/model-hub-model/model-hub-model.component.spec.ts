import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelHubModelComponent } from './model-hub-model.component';

describe('ModelHubModelComponent', () => {
  let component: ModelHubModelComponent;
  let fixture: ComponentFixture<ModelHubModelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ModelHubModelComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ModelHubModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
