module sim_types

    type region
        real (kind=8), pointer, dimension(:,:) :: conserved_var
        integer nconserved_vars
    end type
end module
