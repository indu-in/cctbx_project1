
; Script generated by the HM NIS Edit Script Wizard.
; MUI Settings
!define MUI_ABORTWARNING
!define MUI_ICON "${COPYDIR}\${SOURCEDIR}\modules\gui_resources\icons\custom\WinPhenix.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}${PRODUCT_VERSION}"
!define PRODUCT_ROOT_KEY SHCTX ; replaced with HKLM or HKCU on runtime according to SetShellVarContext
!define PRODUCT_STARTMENU_REGVAL "NSIS:StartMenuDir"

InstallDir "$PROGRAMFILES${IS_64_BIT_PROGRAM}\${PRODUCT_NAME}"

!define MUI_CUSTOMFUNCTION_GUIINIT MyOnGUIinit

!include 'LogicLib.nsh'

; -----------------------
;       UserIsAdmin macro
; -----------------------
;
;   Example:
;       ${If} ${UserIsAdmin}
;           DetailPrint "Current user security context has local administrative rights."
;       ${Else}
;           DetailPrint "Current user security context dose NOT have local administrative rights."
;       ${EndIf}
;
!macro _UserIsAdmin _a _b _t _f
   System::Store 'p0 p1 p2 p3'
   System::Call '*(&i1 0,&i4 0,&i1 5)i.r0'
   System::Call 'advapi32::AllocateAndInitializeSid(i r0,i 2,i 32,i 544,i 0,i 0,i 0,i 0,i 0,i 0,*i .r1)i.r2'
   System::Free $0
   System::Call 'advapi32::CheckTokenMembership(i n,i r1,*i .r2)i.r3'
   IntOp $3 $3 && $2 ; Function success AND was a member
   System::Call 'advapi32::FreeSid(i r1)'

   StrCmp $3 0 0 +3
   ## User is an Admin
     System::Store 'r3 r2 r1 r0'
     Goto `${_f}`

    ## User is not an Admin
     System::Store 'r3 r2 r1 r0'
     Goto `${_t}`
!macroend
!define UserIsAdmin `"" UserIsAdmin ""`



var ALL_OR_USER_TEXT

Function .onInit
  !include x64.nsh
  ${IfNot} ${RunningX64}
    ${If} ${IS_64_BIT_PROGRAM} > 32
  MessageBox MB_ICONEXCLAMATION|MB_OK "This build of ${PRODUCT_NAME} requires a 64 bit version of Windows. \
Please install the 32 bit build of ${PRODUCT_NAME} instead."
  Abort
    ${EndIf}
  ${EndIf}

  ${If} ${UserIsAdmin}
   StrCpy $ALL_OR_USER_TEXT "Setup will install ${PRODUCT_NAME} in the folder below \
for all users. To install in a different folder, click Browse and \
select another folder.$\r$\n$\r$\n\
To install ${PRODUCT_NAME} for just one user exit Setup now. Then run Setup as that user without \
administrative privileges."
   SetShellVarContext all
  ${Else}
   ReadEnvStr $0 HOMEDRIVE
   ReadEnvStr $1 HOMEPATH
   StrCpy $INSTDIR "$0$1\${PRODUCT_NAME}"
   StrCpy $ALL_OR_USER_TEXT "Setup will install ${PRODUCT_NAME} in the folder below. \
To install in a different folder, click Browse and select another folder.$\r$\n$\r$\n\
To install  ${PRODUCT_NAME} for all users exit Setup. Then run Setup with administrative privileges."
   SetShellVarContext current
  ${EndIf}
FunctionEnd


RequestExecutionLevel user ; everyone should be allowed to install phenix to their home folder at least
Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "dist\${PRODUCT_VERSION}\Phenix-${PRODUCT_VERSION}-x${IS_64_BIT_PROGRAM}-Setup.exe"
ShowInstDetails show
ShowUnInstDetails show
AutoCloseWindow false


; MUI 1.67 compatible ------
!include "MUI.nsh"
; Welcome page
!insertmacro MUI_PAGE_WELCOME
; License page
!insertmacro MUI_PAGE_LICENSE "${COPYDIR}\${SOURCEDIR}\LICENSE"
; Components page
!insertmacro MUI_PAGE_COMPONENTS
; Directory page
DirText $ALL_OR_USER_TEXT
!insertmacro MUI_PAGE_DIRECTORY
; Start menu page
var ICONS_GROUP
!define MUI_STARTMENUPAGE_NODISABLE
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "PHENIX"
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "${PRODUCT_ROOT_KEY}"
!define MUI_STARTMENUPAGE_REGISTRY_KEY "${PRODUCT_UNINST_KEY}"
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "${PRODUCT_STARTMENU_REGVAL}"
!insertmacro MUI_PAGE_STARTMENU Application $ICONS_GROUP
; Instfiles page
!insertmacro MUI_PAGE_INSTFILES
; Finish page
!define MUI_FINISHPAGE_SHOWREADME_CHECKED
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\${SOURCEDIR}\README"
!insertmacro MUI_PAGE_FINISH
; Uninstaller pages
!insertmacro MUI_UNPAGE_INSTFILES
; Language files
!insertmacro MUI_LANGUAGE "English"

; MUI end ------


Section "MainSection" SEC01
  SetOutPath "$INSTDIR"
  SetOverwrite off
  File /r /x *.cpp /x *.cc /x *.h /x *.hh /x *.hpp /x *.c /x *.f /x .svn ${COPYDIR}\*

; Shortcuts
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
  !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

Section "Sources" SEC02
  SetOutPath "$INSTDIR"
  SetOverwrite on
  File /nonfatal /r ${COPYDIR}\*.hh
  File /nonfatal /r ${COPYDIR}\*.hpp
  File /nonfatal /r ${COPYDIR}\*.c
  File /nonfatal /r ${COPYDIR}\*.f
  File /nonfatal /r ${COPYDIR}\*.cpp
  File /nonfatal /r ${COPYDIR}\*.cc
  File /nonfatal /r ${COPYDIR}\*.h

; Shortcuts
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
  !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

; Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC01} "Executables, scripts and tutorials"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC02} "C/C++ source files for compilation by expert users"
!insertmacro MUI_FUNCTION_DESCRIPTION_END


Section -AdditionalIcons
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
  WriteIniStr "$INSTDIR\${PRODUCT_NAME}.url" "InternetShortcut" "URL" "${PRODUCT_WEB_SITE}"
  CreateDirectory "$SMPROGRAMS\$ICONS_GROUP\${PRODUCT_VERSION}"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\${PRODUCT_VERSION}\${PRODUCT_NAME}${PRODUCT_VERSION}.lnk" "$INSTDIR\${SOURCEDIR}\build\bin\phenix.bat" "" "${MUI_ICON}"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\${PRODUCT_VERSION}\Documentation.lnk" "$INSTDIR\${SOURCEDIR}\build\bin\phenix.doc.bat" "" "$WINDIR\hh.exe" 0
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\${PRODUCT_VERSION}\Phenix.Python.lnk" "$INSTDIR\${SOURCEDIR}\build\bin\phenix.python.bat" "" "$INSTDIR\${SOURCEDIR}\base\bin\python\python.exe"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\${PRODUCT_VERSION}\Uninstall.lnk" "$INSTDIR\uninst.exe" "" "${MUI_UNICON}"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\PHENIX Website.lnk" "${PRODUCT_WEB_SITE}" "" "$PROGRAMFILES\Internet Explorer\iexplore.exe" 0
  !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

Section -Post
  WriteUninstaller "$INSTDIR\uninst.exe"
  WriteRegStr ${PRODUCT_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "$(^Name)"
  WriteRegStr ${PRODUCT_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninst.exe"
  WriteRegStr ${PRODUCT_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr ${PRODUCT_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
  WriteRegStr ${PRODUCT_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
SectionEnd


Function MyOnGUIinit
!insertmacro UnSelectSection ${SEC01}
!insertmacro UnSelectSection ${SEC02}
FunctionEnd



Function .onSelChange
${If}  ${SectionIsSelected} ${SEC02}
!insertmacro SelectSection ${SEC01}
!insertmacro SetSectionFlag ${SEC01} ${SF_RO}
${Else}
!insertmacro ClearSectionFlag ${SEC01} ${SF_RO}
${EndIf}
FunctionEnd


Function un.onUninstSuccess
  HideWindow
  MessageBox MB_ICONINFORMATION|MB_OK "$(^Name) was successfully removed from your computer."
FunctionEnd

Function un.onInit
  MessageBox MB_ICONQUESTION|MB_YESNO|MB_DEFBUTTON2 "Are you sure you want to completely remove $(^Name) and all of its components?" IDYES +2
  Abort
FunctionEnd

Section Uninstall
  !insertmacro MUI_STARTMENU_GETFOLDER "Application" $ICONS_GROUP

  ;Delete "$SMPROGRAMS\$ICONS_GROUP\Uninstall.lnk"
  ;Delete "$SMPROGRAMS\$ICONS_GROUP\Website.lnk"

  RMDir /r "$SMPROGRAMS\$ICONS_GROUP\${PRODUCT_VERSION}"
  RMDir /r "$INSTDIR\${SOURCEDIR}"

  DeleteRegKey ${PRODUCT_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
  SetAutoClose false
SectionEnd
