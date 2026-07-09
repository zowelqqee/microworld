# Autopilot

Source: https://en.wikipedia.org/wiki/Autopilot
Retrieved at: 2026-06-15T07:16:43Z
Revision ID: 1342840009
Raw text SHA256: 11550916d6a93d5485711138c618ee3f612f98fe9297614ddb7614213fd19406
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
An autopilot is a system used to control the path of an aircraft without requiring constant intervention by a human operator. The autopilot does not replace human operators, but it assists them allowing them to focus on broader aspects of operations (for example, monitoring the trajectory, weather and on-board systems).
When present, an autopilot is often used in conjunction with an autothrottle, a system for controlling the power delivered by the engines.

First autopilots

In the early days of aviation, aircraft required the continuous attention of a pilot to fly safely. As aircraft range increased, allowing flights of many hours, the constant attention led to serious fatigue. An autopilot is designed to perform some of the pilot's tasks.
The first gyroscopic autopilot for aircraft was developed by Sperry Corporation in 1912. The system connected a gyroscopic heading indicator and attitude indicator to hydraulically operated elevators and rudder. It permitted the aircraft to fly straight and level on a compass course without a pilot's attention, greatly reducing the pilot's workload.
Lawrence Sperry, the son of famous inventor Elmer Sperry, demonstrated it in 1914 at an aviation safety contest held in Paris. Sperry demonstrated the credibility of the invention by flying the aircraft with his hands away from the controls and visible to onlookers. Elmer Sperry Jr., the son of Lawrence Sperry, and Capt Shiras continued work on the same autopilot after the war, and in 1930, they tested a more compact and reliable autopilot which kept a U.S. Army Air Corps aircraft on a true heading and altitude for three hours.
In 1930, the Royal Aircraft Establishment in the United Kingdom developed an autopilot called a pilots' assister that used a pneumatically spun gyroscope to move the flight controls.
The autopilot was further developed, to include, for example, improved control algorithms and hydraulic servomechanisms. Adding more instruments, such as radio-navigation aids, made it possible to fly at night and in bad weather. In 1947, a U.S. Air Force C-53 made a transatlantic flight, including takeoff and landing, completely under the control of an autopilot.
Bill Lear developed his F-5 automatic pilot, and automatic approach control system, and was awarded the Collier Trophy in 1949.
In the early 1920s, the Standard Oil tanker J.A. Moffet became the first ship to use an autopilot.
The Piasecki HUP-2 Retriever was the first production helicopter with an autopilot.
The lunar module digital autopilot of the Apollo program is an early example of a fully digital autopilot system in spacecraft.

Modern autopilots

Not all of the passenger aircraft flying today have an autopilot system. Older and smaller general aviation aircraft especially are still hand-flown, and even small airliners with fewer than twenty seats may also be without an autopilot as they are used on short-duration flights with two pilots. The installation of autopilots in aircraft with more than twenty seats is generally made mandatory by international aviation regulations.
There are three levels of control in autopilots for smaller aircraft.

A single-axis autopilot controls an aircraft in the roll axis only; such autopilots are also known colloquially as "wing levellers", reflecting their single capability.
A two-axis autopilot controls an aircraft in the pitch axis as well as roll, and may be little more than a wing leveller with limited pitch oscillation-correcting ability; or it may receive inputs from on-board radio navigation systems to provide true automatic flight guidance once the aircraft has taken off until shortly before landing; or its capabilities may lie somewhere between these two extremes.
A three-axis autopilot adds control in the yaw axis and is not required in many small aircraft.
Autopilots in modern complex aircraft are three-axis and generally divide a flight into taxi, takeoff, climb, cruise (level flight), descent, approach, and landing phases. Autopilots that automate all of these flight phases except taxi and takeoff exist. An autopilot-controlled approach to landing on a runway and controlling the aircraft on rollout (i.e. keeping it on the centre of the runway) is known as an Autoland, where the autopilot utilizes an Instrument Landing System (ILS) Cat IIIc approach, which is used when the visibility is zero. These approaches are available at many major airports' runways today, especially at airports subject to adverse weather phenomena such as fog. The aircraft can typically stop on their own, but will require the disengagement of the autopilot in order to exit the runway and taxi to the gate. An autopilot is often an integral component of a Flight Management System.
Modern autopilots use computer software to control the aircraft. The software reads the aircraft's current position, and then controls a flight control system to guide the aircraft. In such a system, besides classic flight controls, many autopilots incorporate thrust control capabilities that can control throttles to optimize the airspeed.
The autopilot in a modern large aircraft typically reads its position and the aircraft's attitude from an inertial guidance system. Inertial guidance systems accumulate errors over time. They will incorporate error reduction systems such as the carousel system that rotates once a minute so that any errors are dissipated in different directions and have an overall nulling effect. Error in gyroscopes is known as drift. This is due to physical properties within the system, be it mechanical or laser guided, that corrupt positional data. The disagreements between the two are resolved with digital signal processing, most often a six-dimensional Kalman filter. The six dimensions are usually roll, pitch, yaw, altitude, latitude, and longitude. Aircraft may fly routes that have a required performance factor, therefore the amount of error or actual performance factor must be monitored in order to fly those particular routes. The longer the flight, the more error accumulates within the system. Radio aids such as DME, DME updates, and GPS may be used to correct the aircraft position.

Control Wheel Steering

An option midway between fully automated flight and manual flying is Control Wheel Steering (CWS). Although it is becoming less used as a stand-alone option in modern airliners, CWS is still a function on many aircraft today. Generally, an autopilot that is CWS equipped has three positions: off, CWS, and CMD. In CMD (Command) mode the autopilot has full control of the aircraft, and receives its input from either the heading/altitude setting, radio and navaids, or the FMS (Flight Management System). In CWS mode, the pilot controls the autopilot through inputs on the yoke or the stick. These inputs are translated to a specific heading and attitude, which the autopilot will then hold until instructed to do otherwise. This provides stability in pitch and roll. Some aircraft employ a form of CWS even in manual mode, such as the MD-11 which uses a constant CWS in roll. In many ways, a modern Airbus fly-by-wire aircraft in Normal Law is always in CWS mode. The major difference is that in this system the limitations of the aircraft are guarded by the flight control computer, and the pilot cannot steer the aircraft past these limits.

Computer system details
The hardware of an autopilot varies between implementations, but is generally designed with redundancy and reliability as foremost considerations. For example, the Rockwell Collins AFDS-770 Autopilot Flight Director System used on the Boeing 777 uses triplicated FCP-2002 microprocessors which have been formally verified and are fabricated in a radiation-resistant process.
Software and hardware in an autopilot are tightly controlled, and extensive test procedures are put in place.
Some autopilots also use design diversity. In this safety feature, critical software processes will not only run on separate computers, and possibly even using different architectures, but each computer will run software created by different engineering teams, often being programmed in different programming languages. It is generally considered unlikely that different engineering teams will make the same mistakes. As the software becomes more expensive and complex, design diversity is becoming less common because fewer engineering companies can afford it. The flight control computers on the Space Shuttle used this design: there were five computers, four of which redundantly ran identical software, and a fifth backup running software that was developed independently. The software on the fifth system provided only the basic functions needed to fly the Shuttle, further reducing any possible commonality with the software running on the four primary systems.

Stability augmentation systems
A stability augmentation system (SAS) is another type of automatic flight control system; however, instead of maintaining the aircraft required altitude or flight path, the SAS will move the aircraft control surfaces to damp unacceptable motions. SAS automatically stabilizes the aircraft in one or more axes. The most common type of SAS is the yaw damper which is used to reduce the Dutch roll tendency of swept-wing aircraft. Some yaw dampers are part of the autopilot system while others are stand-alone systems.
Yaw dampers use a sensor to detect how fast the aircraft is rotating (either a gyroscope or a pair of accelerometers), a computer/amplifier and an actuator. The sensor detects when the aircraft begins the yawing part of Dutch roll. A computer processes the signal from the sensor to determine the rudder deflection required to damp the motion. The computer tells the actuator to move the rudder in the opposite direction to the motion since the rudder has to oppose the motion to reduce it. The Dutch roll is damped and the aircraft becomes stable about the yaw axis. Because Dutch roll is an instability that is inherent in all swept-wing aircraft, most swept-wing aircraft need some sort of yaw damper.
There are two types of yaw damper: the series yaw damper and the parallel yaw damper. The actuator of a parallel yaw damper will move the rudder independently of the pilot's rudder pedals while the actuator of a series yaw damper is clutched to the rudder control quadrant, and will result in pedal movement when the rudder moves.
Some aircraft have stability augmentation systems that will stabilize the aircraft in more than a single axis. The Boeing B-52, for example, requires both pitch and yaw SAS in order to provide a stable bombing platform. Many helicopters have pitch, roll and yaw SAS systems. Pitch and roll SAS systems operate much the same way as the yaw damper described above; however, instead of damping Dutch roll, they will damp pitch and roll oscillations to improve the overall stability of the aircraft.

Autopilot for ILS landings
Instrument-aided landings are defined in categories by the International Civil Aviation Organization, or ICAO. These are dependent upon the required visibility level and the degree to which the landing can be conducted automatically without input by the pilot.

CAT I – This category permits pilots to land with a decision height of 200 feet (61 m) and a forward visibility or Runway Visual Range (RVR) of 550 metres (1,800 ft). Autopilots are not required.
CAT II – This category permits pilots to land with a decision height between 200 feet (61 m) and 100 feet (30 m) and a RVR of 300 metres (980 ft). Autopilots have a fail passive requirement.
CAT IIIa – This category permits pilots to land with a decision height as low as 50 feet (15 m) and a RVR of 200 metres (660 ft). It needs a fail-passive autopilot. There must be only a 10−6 probability of landing outside the prescribed area.
CAT IIIb – As IIIa but with the addition of automatic roll out after touchdown incorporated with the pilot taking control some distance along the runway. This category permits pilots to land with a decision height less than 50 feet or no decision height and a forward visibility of 250 feet (76 m) in Europe (76 metres, compare this to aircraft size, some of which are now over 70 metres (230 ft) long) or 300 feet (91 m) in the United States. For a landing-without-decision aid, a fail-operational autopilot is needed. For this category some form of runway guidance system is needed: at least fail-passive but it needs to be fail-operational for landing without decision height or for RVR below 100 metres (330 ft).
CAT IIIc – As IIIb but without decision height or visibility minimums, also known as "zero-zero". Not yet implemented as it would require the pilots to taxi in zero-zero visibility. An aircraft that is capable of landing in a CAT IIIb that is equipped with autobrake would be able to fully stop on the runway but would have no ability to taxi.
Fail-passive autopilot: in case of failure, the aircraft stays in a controllable position and the pilot can take control of it to go around or finish landing. It is usually a dual-channel system.
Fail-operational autopilot: in case of a failure below alert height, the approach, flare and landing can still be completed automatically. It is usually a triple-channel system or dual-dual system.

Radio-controlled models
In radio-controlled modelling, and especially RC aircraft and helicopters, an autopilot is usually a set of extra hardware and software that deals with pre-programming the model's flight.

Flight director

A flight director (FD) is a flight instrument that is overlaid on the attitude indicator that shows the pilot of an aircraft the attitude required to execute the desired flight path. While the flight director is separate from the autopilot, they are closely linked. With a flight plan programmed into the flight computer, the flight director will command rolls when turns are required.
Without a flight director, the autopilot is limited to more basic modes, such as maintaining an altitude or a heading, or turning on to a new heading when commanded by the pilot.
When the autopilot and flight director are used together, more complex autopilot modes are possible. The autopilot can follow flight director commands, thus following the flight plan route without pilot intervention.

Nickname
An autopilot system is sometimes colloquially referred to as "George" (e.g. "we'll let George fly for a while"; "George is flying the plane now".). The etymology of the nickname is unclear: some claim it is a reference to American inventor George De Beeson (1897–1965), who patented an autopilot in the 1930s, while others claim that Royal Air Force pilots coined the term during World War II to symbolize that their aircraft technically belonged to King George VI.

See also

Acronyms and abbreviations in avionics
Autonomous aircraft
Uninterruptible autopilot

References

External links

"How Fast Can You Fly Safely", June 1933, Popular Mechanics page 858 photo of Sperry Automatic Pilot and drawing of its basic functions in flight when set
