program test_element_abs

  use element_abs_test, only: dagrt_state_type, &
    timestep_initialize => initialize, &
    timestep_run => run, &
    timestep_shutdown => shutdown

  implicit none

  type(dagrt_state_type), target :: dagrt_state
  type(dagrt_state_type), pointer :: dagrt_state_ptr

  real*8, dimension(100) :: y0

  integer i
  integer stderr
  parameter(stderr=0)

  ! start code ----------------------------------------------------------------

  dagrt_state_ptr => dagrt_state


  do i = 1, 100
    y0(i) = i
  end do

  call timestep_initialize(dagrt_state=dagrt_state_ptr, state_ytype=y0)
  call timestep_run(dagrt_state=dagrt_state_ptr)
  ! For the UserType, check that the absolute value did its job.
  do i = 1, 100
    if (dagrt_state%state_ytype(i) /= 2*y0(i)) then
      write(stderr,*) "UserType elementwise abs failure"
    endif
  end do
  call timestep_shutdown(dagrt_state=dagrt_state_ptr)

end program

