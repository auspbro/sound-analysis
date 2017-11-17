@echo off

:Test_Start
set /a "x=0"
if exist c:\TestProgram\H6C_MCI\SLT1\log\report\port6.wav del c:\TestProgram\H6C_MCI\SLT1\log\report\port6.wav

c:
cd c:\TestProgram\H6C_MCI\SLT1

start /WAIT 1kHz_44100Hz_16bit_10sec.wav

:Start_MBtest
call H6CMTP.exe COM1 AnalogIn6.txt AnalogIn6.bat
ping 1.1.1.1 -n 1 -w 300 >nul
set errorlevel=
find /i "Pass" c:\TestProgram\H6C_MCI\SLT1\log\Report\AnalogIn6.bat
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
echo y | pscp -scp -pw RPM RPM@192.168.0.2:/run/port6.wav c:\TestProgram\H6C_MCI\SLT1\log\report\port6.wav > c:\TestProgram\H6C_MCI\SLT1\log\report\pscplog6.txt
timeout 1
find /i "100%%" c:\TestProgram\H6C_MCI\SLT1\log\report\pscplog6.txt
if not errorlevel 1 goto CheckAnalogIn
set /a "x=x+1"
echo %x% times
if %x%==15 goto end
goto DLAnalogIn

:CheckAnalogIn
if not exist c:\TestProgram\H6C_MCI\SLT1\log\report\port6.wav goto end
python Sound_Analysis\sound_analyzer.py c:\TestProgram\H6C_MCI\SLT1\log\report\port6.wav


:check_left
call c:\TestProgram\H6C_MCI\SLT1\log\Report\port6_out.bat
if "%left%"=="PASS" goto check_right
goto end

:check_right
if "%right%"=="PASS" goto pass
goto end


:pass
cd c:\TestProgram\H6C_MCI\SLT1\Process
>..\log\Test_AnalogIn6_log.bat echo set AnalogIn6=PASS
>>..\log\Test_AnalogIn6_log.bat echo set TestResult=PASS
copy ..\log\Test_AnalogIn6_log.bat ..\logtmp /y
call sdtCheckLog.exe Model_SLT1.cfg AnalogIn6
goto end

:fail
cd c:\TestProgram\H6C_MCI\SLT1\Process
>..\log\Test_AnalogIn6_log.bat echo set AnalogIn6=FAIL
>>..\log\Test_AnalogIn6_log.bat echo set TestResult=FAIL
copy ..\log\Test_AnalogIn6_log.bat ..\logtmp /y
call sdtCheckLog.exe Model_SLT1.cfg AnalogIn6
goto end

:end
taskkill /IM wmplayer.exe