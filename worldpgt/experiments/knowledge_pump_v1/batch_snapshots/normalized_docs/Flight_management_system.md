# Flight management system

Source: https://en.wikipedia.org/wiki/Flight_management_system
Retrieved at: 2026-06-17T17:20:42Z
Revision ID: 1350184263
Raw text SHA256: 1af74be957e3950f1444d20968c388c8ca3394b913a8bd2b9fcca9d108c259f8
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
A flight management system (FMS) is a fundamental component of a modern airliner's avionics. An FMS is a specialized computer system that automates a wide variety of in-flight tasks, reducing the workload on the flight crew to the point that modern civilian aircraft no longer carry flight engineers or navigators. A primary function is in-flight management of the flight plan. Using various sensors (such as GPS and INS often backed up by radio navigation) to determine the aircraft's position, the FMS can guide the aircraft along the flight plan. From the cockpit, the FMS is normally controlled through a control display unit (CDU) that incorporates a small screen and keyboard or touchscreen. The FMS sends the flight plan for display to the electronic flight instrument system (EFIS), navigation display (ND), or multifunction display (MFD). The FMS can be summarised as being a dual system consisting of the flight management computer (FMC), CDU and a cross talk bus.
The modern FMS was introduced on the Boeing 767, though earlier navigation computers existed. Now, systems similar to FMS exist on aircraft as small as the Cessna 182. In its evolution an FMS has had many different sizes, capabilities and controls. However certain characteristics are common to all FMSs.

Navigation database

All FMSs contain a navigation database. The navigation database contains the elements from which the flight plan is constructed. These are defined via the ARINC 424 standard. The navigation database (NDB) is normally updated every 28 days, in order to ensure that its contents are current. Each FMS contains only a subset of the ARINC / AIRAC data, relevant to the capabilities of the FMS.
The NDB contains all of the information required for building a flight plan, consisting of:

Waypoints/Intersection
Airways
Radio navigation aids including distance measuring equipment (DME), VHF omnidirectional range (VOR), non-directional beacons (NDBs) and instrument landing systems (ILSs).
Airports
Runways
Standard instrument departure (SID)
Standard terminal arrival (STAR)
Holding patterns (only as part of IAPs-although can be entered by command of ATC or at pilot's discretion)
Instrument approach procedure (IAP)
Waypoints can also be defined by the pilot(s) along the route or by reference to other waypoints with entry of a place in the form of a waypoint (e.g. a VOR, NDB, ILS, airport or waypoint/intersection).

Flight plan

The flight plan is generally determined on the ground, before departure either by the pilot for smaller aircraft or a professional dispatcher for airliners. It is entered into the FMS either by typing it in, selecting it from a saved library of common routes (Company Routes) or via an ACARS datalink with the airline dispatch center.
During preflight, other information relevant to managing the flight plan is entered. This can include performance information such as gross weight, fuel weight and center of gravity. It will include altitudes including the initial cruise altitude. For aircraft that do not have a GPS, the initial position is also required.
The pilot uses the FMS to modify the flight plan in flight for a variety of reasons. Significant engineering design minimizes the keystrokes in order to minimize pilot workload in flight and eliminate any confusing information (Hazardously Misleading Information).
The FMS also sends the flight plan information for display on the Navigation Display (ND) of the flight deck instruments Electronic Flight Instrument System (EFIS). The flight plan generally appears as a magenta line, with other airports, radio aids and waypoints displayed.
Some FMSs can calculate special flight plans, often for tactical requirements, such as search patterns, rendezvous, in-flight refueling tanker orbits, and calculated air release points (CARP) for accurate parachute jumps.

Position determination
Once in flight, a principal task of the FMS is obtaining a position fix, i.e., to determine the aircraft's position and the accuracy of that position. Simple FMS use a single sensor, generally GPS in order to determine position. But modern FMS use as many sensors as they can, such as VORs, in order to determine and validate their exact position. Some FMS use a Kalman filter to integrate the positions from the various sensors into a single position. Common sensors include:

Airline-quality GPS receivers act as the primary sensor as they have the highest accuracy and integrity.
Radio aids designed for aircraft navigation act as the second highest quality sensors. These include;
Scanning DME (distance measuring equipment) that check the distances from five different DME stations simultaneously in order to determine one position every 10 seconds.
VORs (VHF omnidirectional radio range) that supply a bearing. With two VOR stations the aircraft position can be determined, but the accuracy is limited.
Inertial reference systems (IRS) use ring laser gyros and accelerometers in order to calculate the aircraft position. They are highly accurate and independent of outside sources. Airliners use the weighted average of three independent IRS to determine the “triple mixed IRS” position.
The FMS constantly crosschecks the various sensors and determines a single aircraft position and accuracy. The accuracy is described as the Actual Navigation Performance (ANP) a circle that the aircraft can be anywhere within measured as the diameter in nautical miles.
Modern airspace has a set required navigation performance (RNP). The aircraft must have its ANP less than its RNP in order to operate in certain high-level airspace.

Guidance

Given the flight plan and the aircraft's position, the FMS calculates the course to follow. The pilot can follow this course manually (much like following a VOR radial), or the autopilot can be set to follow the course.
The FMS mode is normally called LNAV or Lateral Navigation for the lateral flight plan and VNAV or vertical navigation for the vertical flight plan. VNAV provides speed and pitch or altitude targets and LNAV provides roll steering command to the autopilot.

VNAV
Sophisticated aircraft, generally airliners such as the Airbus A320 or Boeing 737 and other turbofan powered aircraft, have full performance Vertical Navigation (VNAV). The purpose of VNAV is to predict and optimize the vertical path. Guidance includes control of the pitch axis and control of the throttle.
The FMS needs to have a comprehensive flight and engine model in order to have the data required to do this. The function can create a forecast vertical path along the lateral flight plan using this information. The aircraft manufacturer is usually the only source of this comprehensive flight model.
The vertical profile is constructed by the FMS during pre-flight. Together with the lateral flight plan, it makes use of the aircraft's starting empty weight, fuel weight, center of gravity, and cruising altitude. The first step on a vertical course is to rise to cruise height. Vertical limitations such as "At or ABOVE 8,000" are present in some SID waypoints. Reducing thrust, or "FLEX" climbing, may be used throughout the ascent to spare the engines. Each needs to be taken into account when making vertical profile projections.
Implementation of an accurate VNAV is difficult and expensive, but it pays off in fuel savings primarily in cruise and descent. In cruise, where most of the fuel is burned, there are multiple methods for fuel savings.
As an aircraft burns fuel it gets lighter and can cruise higher where there is less drag. Step climbs or cruise climbs facilitate this. VNAV can determine where the step or cruise climbs (in which the aircraft climbs continuously) should occur to minimize fuel consumption.

Performance optimization allows the FMS to determine the best or most economical speed to fly in level flight. This is often called the ECON speed. This is based on the cost index, which is entered to give a weighting between speed and fuel efficiency. The cost index is calculated by dividing the per-hour cost of operating the plane by the cost of fuel. Generally a cost index of 999 gives ECON speeds as fast as possible without consideration of fuel and a cost index of zero gives maximum fuel economy while disregarding other hourly costs such as maintenance and crew expenses. ECON mode is the VNAV speed used by most airliners in cruise.
RTA or required time of arrival allows the VNAV system to target arrival at a particular waypoint at a defined time. This is often useful for airport arrival slot scheduling. In this case, VNAV regulates the cruise speed or cost index to ensure the RTA is met.
The first thing the VNAV calculates for the descent is the top of descent point (TOD). This is the point where an efficient and comfortable descent begins. Normally this will involve an idle descent, but for some aircraft an idle descent is too steep and uncomfortable. The FMS calculates the TOD by “flying” the descent backwards from touchdown through the approach and up to cruise. It does this using the flight plan, the aircraft flight model and descent winds. For airline FMS, this is a very sophisticated and accurate prediction, for simple FMS (on smaller aircraft) it can be determined by a “rule of thumb” such as a 3 degree descent path.
From the TOD, the VNAV determines a four-dimensional predicted path. As the VNAV commands the throttles to idle, the aircraft begins its descent along the VNAV path. If either the predicted path is incorrect or the downpath winds different from the predictions, then the aircraft will not perfectly follow the path. The aircraft varies the pitch in order to maintain the path. Since the throttles are at idle this will modulate the speed. Normally the FMS allows the speed to vary within a small band. After this, either the throttles advance (if the aircraft is below path) or the FMS requests speed brakes with a message, often "DRAG REQUIRED" (if the aircraft is above path). On Airbus aircraft, this message also appears on the PFD and, if the aircraft is extremely high on path, "MORE DRAG" will be displayed. On Boeing aircraft, if the aircraft gets too far off the prescribed path, it will switch from VNAV PTH (which follows the calculated path) to VNAV SPD (which descends as fast as possible while maintaining a selected speed, similar to OP DES (open descent) on Airbuses.
An ideal idle descent, also known as a “green descent” uses the minimum fuel, minimizes pollution (both at high altitude and local to the airport) and minimizes local noise. While most modern FMS of large airliners are capable of idle descents, most air traffic control systems cannot handle multiple aircraft each using its own optimum descent path to the airport, at this time. Thus the use of idle descents is minimized by Air Traffic Control.

See also
Index of aviation articles
Acronyms and abbreviations in avionics
Strategic Lateral Offset Procedure

References

Further reading
ARINC 702A, Advanced Flight Management Computer System
Avionics, Element, Software and Functions Ch 20, Cary R. Spitzer, ISBN 0-8493-8438-9
FMC User's Guide B737, Ch 1, Bill Bulfer, Leading Edge Libraries
Casner, S.M. The Pilot's Guide to the Modern Airline Cockpit. Newcastle WA, Aviation Supplies and Academics, 2007. ISBN 1-56027-683-5.
Chappell, A.R. et al. "The VNAV Tutor: Addressing a Mode Awareness Difficulty for Pilots of Glass Cockpit Aircraft." IEEE Transactions on Systems, Man, and Cybernetics Part A, Systems and Humans, vol. 27, no.3, May 1997, pp. 372–385.
