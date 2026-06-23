# Security and Maintenance

Source: https://en.wikipedia.org/wiki/Security_and_Maintenance
Retrieved at: 2026-06-22T08:08:21Z
Revision ID: 1336017762
Raw text SHA256: f7887d08d4192a5d5b2c6a8f7ef101a73e710009657a83d865afe53083fda2e4
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
Security and Maintenance (formerly known as Action Center, and Security Center in earlier versions) is a component of the Windows NT family of operating systems that monitors the security and maintenance status of the computer. Its monitoring criteria includes optimal operation of antivirus software, personal firewall, as well as the working status of Backup and Restore, Network Access Protection (NAP), User Account Control (UAC), Windows Error Reporting (WER), and Windows Update. It notifies the user of any problem with the monitored criteria, such as when an antivirus program is not up-to-date or is offline.

Operation
Security and Maintenance consists of three major components: A control panel applet, a Windows service and an application programming interface (API) provided by Windows Management Instrumentation (WMI).
The control panel applet divides the monitored criteria into categories and color-codes them. Yellow indicates a non-critical warning, e.g. some settings are not being monitored or are not optimal. Red indicates a critical message, e.g. anti-virus program is offline.
A service, named "Security Center", determines the current state of the settings. The service, by default, starts when the computer starts; it continually monitors the system for changes, and notifies the user if it detects a problem. In versions of Windows prior to Windows 10, it adds a notification icon into the Windows Taskbar.
A WMI provider makes the settings available to the system. Third-party anti-virus, anti-spyware and personal firewall software vendors primarily register with Security and Maintenance through the WMI provider. Windows Vista added a new set of APIs that let programs retrieve the aggregate health status within Security and Maintenance, and to receive notifications when the health status changes. These APIs allow programs to confirm that the system is in a healthy state before engaging in certain actions. For example, a computer game can ensure that a firewall is running before connecting to an online game.
Security and Maintenance is in charge of the following:

Querying the status of the personal firewall and turning it on
Querying the status of the anti-malware program, turning it on and instructing it to update itself
Querying the status of the Internet security settings and asking the user to change them if they are not optimal
Querying the status of the User Account Control settings and asking the user to change it if it is not optimal
Scheduling and executing automatic maintenance tasks, which includes a quick scan for malware, disk defragmentation, power efficiency diagnostics
Querying the status of Backup and Restore and prompting the user to schedule a backup if one is not in place (Windows 7 only)
Querying the status of File History; however, the user is not alerted about it (Windows 8 and later only)
Querying the status of HomeGroup; no alerts are issued about it
Managing problems logged by Windows Error Reporting: The user can see their details, send them to Microsoft if they are not automatically sent, query a solution for them (although most of the time, there is none) or selectively delete them.

Version history

Windows XP SP2
Microsoft learned from discussions with customers that there was confusion as to whether users were taking appropriate steps to protect their systems, or if the steps they were taking were effective. From this research, Microsoft made the decision to include a visible control panel with Windows XP Service Pack 2 that would provide a consolidated view of the most important security features. Service Pack 2, released in August 2004, includes the first version of Windows Security Center (WSC). This version monitors Windows Update, Windows Firewall, and the availability of an anti-virus program. Third-party providers of personal firewall and anti-virus software packages were encouraged to use WSC API to register their products with WSC.
On August 25, 2004, PC Magazine published an article in their Security Watch newsletter titled "Windows XP SP2 Security Center Spoofing Threat" which outlined a design vulnerability which could allow malware to manipulate Security Center into displaying a false security status regardless of the true security status. To do so, the malware requires administrative privileges. Microsoft countered their claim by asserting that if a piece of malware gains administrative privileges, it need not spoof anything, as it can commit much nastier malicious actions.

Windows Vista
WSC in Windows Vista monitors new criteria, such as anti-spyware software, User Account Control, and Internet Explorer security settings. It can also display logos of third-party products that have been registered with the Security Center.
Unlike Windows XP, in the beta versions of Windows Vista, WSC could not be disabled or overridden. Security software maker Symantec spoke out against this, noting that it would cause a great deal of consumer confusion because any security problems would be reported by both WSC and Symantec's tools at the same time. McAfee, another large security software vendor, lodged similar complaints. In the end, Microsoft allowed WSC to be disabled.

Windows 7
In Windows 7, Windows Security Center has been renamed Action Center. It was designed to centralize and reduce the number of notifications about the system; as such, it encompasses both security and maintenance of the computer. Its notification icon on Windows Taskbar only appears when there is a message for perusal and replaces five separate notification icons found in Windows Vista. A "Troubleshooting" link was also added, providing a shortcut to Windows 7's new Troubleshooting control panel.

Windows 8
In Windows 8, Action Center monitors 10 new items: Microsoft account, Windows activation, SmartScreen, automatic maintenance, drive status, device software, startup apps, HomeGroup, File History, and Storage Spaces.

Windows 10

In Windows 10, the name "Action Center" is now used for application notifications and quick actions. The Action Center from Windows 8.1 was renamed to Security and Maintenance, causing confusion for users and IT administrators. It no longer displays an icon in the notification area, but otherwise retains all the features of the Windows 8.1 Action Center. The "Troubleshooting" link was removed in Windows 10 Fall Creators Update.
Starting with Windows 10 Creators Update, Microsoft has introduced a new component called Windows Defender Security Center (WDSC) that provides much of the same functionality. This new component is a Universal Windows Platform app and is also the default front-end for Windows Defender. It relies on its own service, called "Windows Defender Security Center Service".
In comparison to Security and Maintenance, the WDSC:

monitors antivirus and firewall software, device drivers, device security, storage capacity, account protection, parental control, SmartScreen and Windows Update
has its own distinct icon in the notification area (a black-and-white shield divided by a cross in four sectors)
can fully control Windows Defender
recognizes and supports third-party antivirus and firewall (version 1709 and later): upon detecting such software, it will automatically disable itself
In Windows 10 version 1809, the Windows Defender Security Center was renamed to Windows Security.

See also
List of Microsoft Windows components
Microsoft Defender Antivirus
Microsoft Security Essentials

References

External links
MSDN: Windows Security Center API
