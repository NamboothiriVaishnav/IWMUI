import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AutoMlComponent } from './auto-ml.component';

describe('AutoMlComponent', () => {
  let component: AutoMlComponent;
  let fixture: ComponentFixture<AutoMlComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AutoMlComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(AutoMlComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
