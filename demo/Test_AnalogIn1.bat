@echo off

:Test_Start
set /a "x=0"
if exist c:\TestProgram\H6C_MCI\SLT1\log\report\port1.wav del c:\TestProgram\H6C_MCI\SLT1\log\report\port1.wav

c:
cd c:\TestProgram\H6C_MCI\SLT1

start /WAIT 1kHz_44100Hz_16bit_10sec.wav

:Start_MBtest
call H6CMTP.exe COM1 AnalogIn1.txt AnalogIn1.bat
ping 1.1.1.1 -n 1 -w 300 >nul
set errorlevel=
find /i "Pass" c:\TestProgram\H6C_MCI\SLT1\log\Report\AnalogIn1.bat
if not errorlevel 1 goto AnalogInOff
goto end

:AnalogInOff
call H6CMTP.exe COM1 AnalogInOff.txt AnalogInOff.bat
ping 1.1.1.1 -n 1 -w 300 >nul
set errorlevel=
find /i "Pass" c:\TestProgram\H6C_MCI\SLT1\log\Report\AnalogInOff.bat
if not errorlevel 1 goto DLAnalogIn
goto end

:DLAnalogIn
echo y | pscp -scp -pw RPM RPM@192.168.0.2:/run/port1.wav c:\TestProgram\H6C_MCI\SLT1\log\report\port1.wav > c:\TestProgram\H6C_MCI\SLT1\log\report\pscplog1.txt
timeout 1
find /i "100%%" c:\TestProgram\H6C_MCI\SLT1\log\report\pscplog1.txt
if not errorlevel 1 goto CheckAnalogIn
set /a "x=x+1"
echo %x% times
if %x%==15 goto end
goto DLAnalogIn

:CheckAnalogIn
if not exist c:\TestProgram\H6C_MCI\SLT1\log\report\port1.wav goto end
python Sound_Analysis\sound_analyzer.py c:\TestProgram\H6C_MCI\SLT1\log\report\port1.wav


:check_left
call c:\TestProgram\H6C_MCI\SLT1\log\Report\port1_out.bat
if "%left%"=="PASS" goto check_right
goto end

:check_right
if "%right%"=="PASS" goto pass
goto end


rem A85Audio_Test.exe mci c:\TestProgram\H6C_MCI\SLT1\log\report\port0.wav out_AnalogIn0
::ren c:\TestProgram\H6C_MCI\SLT1\log\report\out_AnalogIn4.txt out_AnalogIn0.bat
find /i "Pass" c:\TestProgram\H6C_MCI\SLT1\log\Report\out_AnalogIn0.txt
if not errorlevel 1 goto pass
goto end

:pass
cd c:\TestProgram\H6C_MCI\SLT1\Process
>..\log\Test_AnalogIn1_log.bat echo set AnalogIn1=PASS
>>..\log\Test_AnalogIn1_log.bat echo set TestResult=PASS
copy ..\log\Test_AnalogIn1_log.bat ..\logtmp /y
call sdtCheckLog.exe Model_SLT1.cfg AnalogIn1
goto end

:fail
cd c:\TestProgram\H6C_MCI\SLT1\Process
>..\log\Test_AnalogIn1_log.bat echo set AnalogIn1=FAIL
>>..\log\Test_AnalogIn1_log.bat echo set TestResult=FAIL
copy ..\log\Test_AnalogIn1_log.bat ..\logtmp /y
call sdtCheckLog.exe Model_SLT1.cfg AnalogIn1
goto end

:end