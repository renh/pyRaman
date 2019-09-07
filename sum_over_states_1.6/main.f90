program main
    
    use prec
    use fdf
    
    implicit none

    real(q),dimension(:),allocatable :: freq
    real(q),dimension(:),allocatable :: D
    integer :: N_modes, N_gs, N_es, f_gs, r_es
    character*32 :: fname
    integer :: ios
    integer :: i, j, m, n, r
    integer :: f1, f2, r1, r2
    real(q),dimension(:,:,:),allocatable :: S
    real(q) :: gamma
    complex(q) :: dump
    complex(q),dimension(:,:),allocatable :: alpha
    complex(q),dimension(:,:),allocatable :: mix_alpha
    complex(q) :: mix_dump
    real(q),dimension(:,:),allocatable :: intensity
    real(q),dimension(:,:),allocatable :: mix_intensity
    real(q) :: Ex_energy, In_energy, dipole


    call fdf_init('STDIN','INPUT_bak.fdf')
    N_gs = fdf_integer("NumberOfVibStatesGround", 10)
    N_es = fdf_integer("NumberOfVibStatesExcited", 20)
    fname = fdf_string("DisplacementFile",'disp.dat')
    gamma = fdf_physical("Broadening",100.0_q,'cm-1')

    if (.not. fdf_defined('ExcitationEnergy')) then
        write(6,*) 'Error: Electronic excitation energy required.'
        stop
    endif
    Ex_energy = fdf_physical('ExcitationEnergy',0._q,'cm-1')
    write(6,*) "Excitation energy in cm-1:", Ex_energy

    if (.not. fdf_defined("IncidentEnergy")) then
        write(6,*) 'Error: Incident energy required.'
        stop
    endif
    In_energy = fdf_physical('IncidentEnergy', 0._q, 'cm-1')
    write(6,*) "Incident energy in cm-1: ", In_energy 

    if (.not. fdf_defined("TransitionDipole")) then
        write(6,*) "Error: Transition dipole required."
        stop
    endif
    dipole = fdf_physical("TransitionDipole", 0._q, 'e*Bohr')
    write(6,*) "Transition Dipole (au):", dipole

    !count how many vib modes
    N_modes = 0
    open(21,file=fname)
    do
        read(21,*,iostat=ios)
        if (ios /= 0) exit
        N_modes = N_modes + 1
    enddo
    close(21)
    write(6,*) "Number of modes: ", N_modes

    allocate(freq(N_modes))
    allocate(D(N_modes))
    allocate(S(N_gs, N_es, N_modes))
!    allocate(dump(N_modes))
    allocate(alpha(N_gs,N_modes))
    allocate(intensity(N_gs,N_modes))

    allocate(mix_alpha(N_modes,N_modes))
    allocate(mix_intensity(N_modes,N_modes))
    
    ! Read dimensionless displacements and freq
    open(21,file=fname)
    do i = 1, N_modes
        read(21,*) freq(i), D(i)
        ! Now freq are in cm-1
    enddo
    close(21)

    ! assign values to S matrix
    S = 0.0_q
    S(1,1,:) = exp(-.5_q*D*D)
    do i = 2, N_gs
        S(i,1,:) = -1._q * D / (i-1)**.5_q * S(i-1,1,:)
    enddo
    do i = 2, N_es
        S(1,i,:) = D / (i-1)**.5_q * S(1,i-1,:)
    enddo

    do i = 2, N_es
        do j = 2, N_gs
            S(j,i,:) = real((i-1)/j,q) ** .5_q * S(j-1,i-1,:) - D * &
            &   1._q/real(j,q)**.5_q * S(j-1,i,:)
        enddo
    enddo
    write(6,*) "Franck-Condon integrals done."

    alpha = (0._q,0._q)
    mix_alpha = (0._q,0._q)

    do f_gs = 2, N_gs
        do r_es = 1, N_es
        do i = 1, N_modes
            dump = S(f_gs,r_es,i) * S(1,r_es,i) / &
                ((r_es-1)*freq(i) + Ex_energy - In_energy - (0._q,1._q)*gamma) + &
                S(f_gs,r_es,i) * S(1,r_es,i) / &
                ((r_es-1)*freq(i) + Ex_energy + In_energy + (0._q,1._q)*gamma)
            alpha(f_gs,i) = alpha(f_gs,i) + dump
        enddo
        enddo
    enddo
    !print *, alpha(2,1)*conjg(alpha(2,1))
    !print *, alpha(2,2)*conjg(alpha(2,2))

    intensity = alpha * conjg(alpha) * dipole * dipole

    ! Calculate overtones contributed from mixed vib modes
    f1 = 2 ! |f1> = |1>
    f2 = 2 ! |f2> = |1>
    do i = 1, N_modes
    do j = i + 1, N_modes
        do r1 = 1, N_es
        do r2 = 1, N_es
            mix_dump = S(f1,r1,i)*S(f2,r2,j)*S(1,r1,i)*S(1,r2,j) * &
                (1._q/ (Ex_energy + (r1-1)*freq(i) + (r2-1)*freq(j) - In_energy - & 
                    (0._q,1._q)*gamma ) + &
                (1._q/ (Ex_energy + (r1-1)*freq(i) + (r2-1)*freq(j) + In_energy + &
                    (0._q,1._q)*gamma )) )
            mix_alpha(i,j) = mix_alpha(i,j) + mix_dump
        enddo
        enddo
        !print *, mix_alpha(i,j)*conjg(mix_alpha(i,j))
        mix_intensity(i,j) = mix_alpha(i,j) * conjg(mix_alpha(i,j)) * dipole *&
            dipole
    enddo
    enddo

!=========================================================================
! Writeout intensities
!=========================================================================

    ! All contributions, from fundamental and overtones
    open(31,file='intensity.dat')
    do f_gs = 2, N_gs
        do i = 1, N_modes
            write(31,"(f12.4,E18.8)") (f_gs-1)*freq(i), & 
                intensity(f_gs,i) * In_energy**3 * (In_energy-(f_gs-1)*freq(i))
        enddo
    enddo

    open(35,file="mix.dat")
    do i = 1, N_modes
        do j = i+1, N_modes
            write(31,"(f12.4,E18.8)") freq(i)+freq(j), &
                mix_intensity(i,j) * IN_energy**3 * (IN_energy-freq(i)-freq(j))
            write(35,"(f12.4,E18.8,f12.4,f12.4)") freq(i)+freq(j), &
                mix_intensity(i,j)*IN_energy**3 *(IN_energy-freq(i)-freq(j)), &
                freq(i), freq(j)
        enddo
    enddo
    close(31)
    close(35)

    ! All fundamentals
    open(32,file="fundamental.dat")
    do i = 1, N_modes
        write(32,"(f12.4,E18.8)") freq(i), &
            intensity(2,i) * In_energy**3 * (In_energy-freq(i))
    enddo
    close(32)

    open(33,file="double.dat")
    do i = 1, N_modes
        write(33,"(f12.4,E18.8)") 2*freq(i), &
            intensity(3,i) * In_energy**3 * (In_energy-2*freq(i))
    enddo
    close(33)

    open(34,file="higher.dat")
    do f_gs = 4, N_gs
        do i = 1, N_modes
            write(34,"(f12.4,E18.8)")  (f_gs-1) * freq(i), &
                intensity(f_gs,i) * In_energy**3 * (In_energy-(f_gs-1)*freq(i))
        enddo
    enddo
    close(34)

    deallocate(freq)
    deallocate(D)
    deallocate(S)
!    deallocate(dump)
    deallocate(alpha)
    deallocate(mix_alpha)


end program main
