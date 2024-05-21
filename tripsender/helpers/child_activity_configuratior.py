import json

def collect_data_for_group():
    age_group = input("Enter age group (comma-separated): ").split(',')
    age_group = [int(age.strip()) for age in age_group]

    mandatory_activities = []
    activities_input = input("Enter mandatory activities (comma-separated): ").split(',')
    if len(activities_input) > 1:
        for activity in activities_input:
            prob = float(input(f"Enter probability for {activity.strip()}: "))
            mandatory_activities.append({"name": activity.strip(), "probability": prob})
    else:
        mandatory_activities.append({"name": activities_input[0].strip(), "probability": 1})

    duration = int(input("Enter duration (in hours): "))

    after_school_activities = []
    while True:
        activity = input("Enter an after-school activity (or 'done' to finish): ")
        if activity.lower() == 'done':
            break
        duration_str = input("Duration (in minutes): ").zfill(4)
        variance_str = input("Variance (in minutes): ").zfill(4)
        after_school_activities.append({"activity": activity, "duration": duration_str, "variance": variance_str})

    return {
        "age": age_group,
        "mandatory_activity": mandatory_activities,
        "duration": duration,
        "after_school": after_school_activities
    }

def create_children_activity_json():
    data = {"activities": []}
    unassigned_ages = set(range(1, 18))
    
    while unassigned_ages:
        print(f"\nUnassigned ages: {sorted(list(unassigned_ages))}\n")
        group_data = collect_data_for_group()
        
        # Check if ages are valid and unassigned
        if all(age in unassigned_ages for age in group_data['age']):
            data["activities"].append(group_data)
            unassigned_ages.difference_update(group_data['age'])
        else:
            print("\nSome of the ages are either assigned already or out of range. Please enter again.")

    # Save to JSON
    with open('CHILDREN_ACTIVITY.json', 'w') as f:
        json.dump(data, f, indent=4)

    print("\nData saved to 'children_activity.json'")

# Run
create_children_activity_json()
