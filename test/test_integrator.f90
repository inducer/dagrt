program test_rkmethod
  use RKMethod, only: leap_state_type, initialize, shutdown, &
    leap_state_func_initialization, &
    leap_state_func_primary

  implicit none

  type(leap_state_type), target :: state
  type(leap_state_type), pointer :: state_ptr

  real*8 initial_condition(2)

  integer i

  ! start code ----------------------------------------------------------------

  state_ptr => state

  initial_condition(1) = 2
  initial_condition(2) = 2

  call initialize( &
    leap_state=state_ptr, &
    state_y=initial_condition, &
    leap_t=0d0, &
    leap_dt=1d-4)

  call leap_state_func_initialization(leap_state=state_ptr)
  do i = 1,3
    call leap_state_func_primary(leap_state=state_ptr)
    write(*,*) state%ret_state_y
  enddo

  call shutdown(leap_state=state_ptr)

end program
